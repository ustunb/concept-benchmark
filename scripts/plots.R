#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(stringr)
  library(scales)
})

option_list <- list(
  make_option("--run_name", type="character"),
  make_option("--match_mode", type="character", default="substring"),
  make_option("--results_dir", type="character", default=NULL),
  make_option("--out_subdir", type="character", default="demo_images"),
  make_option("--expected_budgets", type="character", default="0,1,2,5,10"),
  make_option("--agg", type="character", default="mean"),
  make_option("--cbm_best_keys", type="character", default="intervene100"),
  make_option("--cbm_ma_keys", type="character", default="intervene100,lfcbm"),
  make_option("--cbm_expert_keys", type="character", default="intervene70"),
  make_option("--cbm_subjective_keys", type="character", default="intervene70,noise30"),
  make_option("--per_concept_k", type="integer", default=10),
  make_option("--table_budget", type="integer", default=1)   # <— NEW: default k for the table
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$run_name) || nchar(opt$run_name) == 0) stop("run_name is required", call. = FALSE)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)])
script_dir <- if (length(script_path) == 0) getwd() else dirname(script_path)

results_dir <- if (!is.null(opt$results_dir)) {
  normalizePath(opt$results_dir, mustWork = TRUE)
} else {
  normalizePath(file.path(script_dir, "..", "results"), mustWork = TRUE)
}
robot_roots <- c(file.path(results_dir, "robot_text"), file.path(script_dir, "robot_text"))
robot_roots <- robot_roots[dir.exists(robot_roots)]
if (length(robot_roots) == 0) stop("No robot_text directory under ../results or scripts/", call. = FALSE)

out_dir <- file.path(results_dir, opt$out_subdir, opt$run_name)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

exp_budgets <- str_split(opt$expected_budgets, ",", simplify = TRUE) %>% as.integer()
agg_fun <- if (tolower(opt$agg) == "median") \(x) median(x, na.rm = TRUE) else \(x) mean(x, na.rm = TRUE)

list_runs <- function(root) {
  tibble(path = list.dirs(root, full.names = TRUE, recursive = FALSE)) %>%
    mutate(rundir = basename(path))
}
runs <- map_dfr(robot_roots, list_runs) %>%
  filter(str_detect(rundir, "^(best|expert|subjective)_cbm_")) %>%
  filter(!str_detect(rundir, "anchor"))

if (tolower(opt$match_mode) == "exact") {
  runs <- runs %>% filter(rundir == opt$run_name)
} else {
  runs <- runs %>% filter(str_detect(rundir, fixed(opt$run_name)))
}
if (nrow(runs) == 0) stop("No CBM run folders matched run_name under robot_text", call. = FALSE)

split_keys <- function(s) {
  ks <- str_split(s, ",", simplify = TRUE) %>% as.vector()
  trimws(ks[ks != ""])
}
req_best       <- split_keys(opt$cbm_best_keys)
req_ma         <- split_keys(opt$cbm_ma_keys)
req_expert     <- split_keys(opt$cbm_expert_keys)
req_subjective <- split_keys(opt$cbm_subjective_keys)

has_all <- function(names, keys) {
  if (length(keys) == 0) return(rep(TRUE, length(names)))
  out <- rep(TRUE, length(names))
  for (k in keys) out <- out & str_detect(names, fixed(k))  # strict substring match; no fuzzy intervene/noise
  out
}

runs <- runs %>%
  mutate(
    orig_regime = str_match(rundir, "^(best|expert|subjective)_cbm_")[,2],
    is_ma = orig_regime == "best" & str_detect(rundir, fixed("lfcbm"))
  ) %>%
  filter(
    (orig_regime == "best"       & !is_ma & has_all(rundir, req_best))       |
    (orig_regime == "best"       &  is_ma & has_all(rundir, req_ma))         |
    (orig_regime == "expert"     &         has_all(rundir, req_expert))      |
    (orig_regime == "subjective" &         has_all(rundir, req_subjective))
  ) %>%
  mutate(regime = if_else(is_ma, "ma", orig_regime)) %>%
  select(path, rundir, regime)

if (nrow(runs) == 0) stop("After regime keyword gates, no CBM runs remain. Adjust --cbm_*_keys.", call. = FALSE)

pick_viability <- function(run_path) {
  cand <- list.files(run_path, pattern = "viability_robots_text_complete_.*_(detected|machine)\\.csv$", full.names = TRUE)
  if (length(cand) == 0) return(NA_character_)
  cand[order(file.info(cand)$mtime, decreasing = TRUE)][1]
}
pick_preds <- function(run_path) {
  cand <- list.files(run_path, pattern = "preds_test_k0_k10_complete_.*_(detected|machine)\\.csv$", full.names = TRUE)
  if (length(cand) == 0) return(NA_character_)
  cand[order(file.info(cand)$mtime, decreasing = TRUE)][1]
}
pick_per_concept <- function(run_path, k) {
  pat <- sprintf("interventions_per_concept_complete_.*_(detected|machine)_taraw_k%d\\.csv$", k)
  cand <- list.files(run_path, pattern = pat, full.names = TRUE)
  if (length(cand) == 0) return(NA_character_)
  cand[order(file.info(cand)$mtime, decreasing = TRUE)][1]
}

runs <- runs %>%
  group_by(regime) %>%
  mutate(
    viab = pick_viability(path),
    preds = pick_preds(path),
    per_concept = pick_per_concept(path, opt$per_concept_k),
    per_concept_k0 = pick_per_concept(path, 0)
  ) %>%
  slice_head(n = 1) %>% ungroup() %>%
  filter(!is.na(viab))

if (nrow(runs) == 0) stop("No viability CSVs found in filtered runs.", call. = FALSE)

pick_col <- function(nms, cands) {
  for (c in cands) if (c %in% nms) return(c)
  NA_character_
}
find_first_regex <- function(nms, pats) {
  for (p in pats) {
    idx <- which(grepl(p, nms, ignore.case = TRUE))[1]
    if (!is.na(idx)) return(nms[idx])
  }
  NA_character_
}

# ---------- Loud-fail helpers for the table path ----------
fail <- function(...) stop(paste0(...), call. = FALSE)

validate_viability_for_table <- function(viab_path, regime, table_k) {
  if (is.na(viab_path) || !file.exists(viab_path)) {
    fail("No viability CSV for regime=", regime, " (", viab_path, ").")
  }
  df <- suppressMessages(readr::read_csv(viab_path, show_col_types = FALSE))
  if (!("budget" %in% names(df))) {
    if ("k" %in% names(df)) df <- dplyr::rename(df, budget = k) else {
      fail("Missing required column budget (or k) in ", viab_path, ".")
    }
  }
  if ("concept_source" %in% names(df)) {
    want <- if (regime == "ma") "machine" else "detected"
    df <- df %>% dplyr::filter(concept_source == want)
    if (nrow(df) == 0) fail("concept_source filter removed all rows in ", viab_path, " for regime=", regime, ".")
  }
  if ("target_acc" %in% names(df)) {
    df <- df %>% dplyr::filter(target_acc == "raw")
    if (nrow(df) == 0) fail("No rows with target_acc=='raw' in ", viab_path, ".")
  }
  if (!("raw_gain_vs_k0" %in% names(df))) {
    fail("viability missing required column raw_gain_vs_k0: ", viab_path, ".")
  }
  have <- sort(unique(as.integer(df$budget)))
  if (!(table_k %in% have)) {
    fail("viability missing required budget k=", table_k, " in ", viab_path,
         ". Present budgets: ", paste(have, collapse=","), ".")
  }
  invisible(TRUE)
}
# ---------------------------------------------------------

read_viab_cbm <- function(viab_path, regime) {
  df0 <- suppressMessages(readr::read_csv(viab_path, show_col_types = FALSE))
  if (!("budget" %in% names(df0))) {
    if ("k" %in% names(df0)) df0 <- dplyr::rename(df0, budget = k) else stop("Missing budget in ", viab_path)
  }
  if ("concept_source" %in% names(df0)) {
    want <- if (regime == "ma") "machine" else "detected"
    df0 <- df0 %>% filter(concept_source == want)
  }
  if ("target_acc" %in% names(df0)) df0 <- df0 %>% filter(target_acc == "raw")
  post_col <- pick_col(names(df0), c("acc_cbm_intv","acc_cbm_post","acc_post","sel_acc_post"))
  present <- sort(intersect(unique(df0$budget) %>% as.integer(), exp_budgets))
  if (length(present) == 0) return(tibble())
  df1 <- df0 %>%
    filter(budget %in% present) %>%
    group_by(budget) %>%
    summarize(
      acc_post = agg_fun(.data[[post_col]]),
      raw_gain_vs_k0 = agg_fun(if ("raw_gain_vs_k0" %in% names(cur_data())) .data[["raw_gain_vs_k0"]] else NA_real_),
      delta_vs_blackbox = agg_fun(if ("delta_vs_blackbox" %in% names(cur_data())) .data[["delta_vs_blackbox"]] else if ("gain_acc_dnn" %in% names(cur_data())) .data[["gain_acc_dnn"]] else NA_real_),
      concept_checks = agg_fun(if ("concept_checks" %in% names(cur_data())) .data[["concept_checks"]] else NA_real_),
      interventions_total = agg_fun(if ("interventions_total" %in% names(cur_data())) .data[["interventions_total"]] else NA_real_),
      attempted_edits_total = agg_fun(if ("attempted_edits_total" %in% names(cur_data())) .data[["attempted_edits_total"]] else NA_real_),
      corrected_edits_total = agg_fun(if ("corrected_edits_total" %in% names(cur_data())) .data[["corrected_edits_total"]] else NA_real_),
      concepts_per_intervention = agg_fun(if ("concepts_per_intervention" %in% names(cur_data())) .data[["concepts_per_intervention"]] else NA_real_),
      applied_edits_total = agg_fun(if ("applied_edits_total" %in% names(cur_data())) .data[["applied_edits_total"]] else NA_real_),
      avg_edits_per_case = agg_fun(if ("avg_edits_per_case" %in% names(cur_data())) .data[["avg_edits_per_case"]] else NA_real_),
      failed_interventions_pct = agg_fun(if ("failed_interventions_pct" %in% names(cur_data())) .data[["failed_interventions_pct"]] else NA_real_),
      coverage_after_confirmation = agg_fun(if ("coverage_after_confirmation" %in% names(cur_data())) .data[["coverage_after_confirmation"]] else NA_real_),
      .groups = "drop"
    ) %>%
    mutate(system = "CBM", regime = regime) %>%
    relocate(system, regime, budget)
  df1
}

tidy <- pmap_dfr(list(runs$viab, runs$regime), read_viab_cbm) %>%
  arrange(regime, budget)

common_budgets <- tidy %>% group_by(regime) %>% summarize(b = list(unique(budget))) %>% pull(b) %>% reduce(intersect)
tidyK <- tidy %>% filter(budget %in% common_budgets)

get_n <- function(preds_path) {
  if (is.na(preds_path)) return(NA_integer_)
  suppressMessages(readr::read_csv(preds_path, show_col_types = FALSE)) %>% nrow()
}
n_by_regime <- map_int(runs$preds, get_n)
names(n_by_regime) <- runs$regime

tidyK <- tidyK %>% mutate(
  N = n_by_regime[regime],
  checks_per_case = ifelse(!is.na(N) & N > 0, concept_checks / N, NA_real_),
  confirm_cost_per_intervention = ifelse(!is.na(interventions_total) & interventions_total > 0, concept_checks / interventions_total, NA_real_),
  edit_success_rate = ifelse(!is.na(attempted_edits_total) & attempted_edits_total > 0, corrected_edits_total / attempted_edits_total, NA_real_),
  regime_label = factor(
    case_when(
      regime == "best" ~ "Best",
      regime == "expert" ~ "Expert",
      regime == "subjective" ~ "Subjective",
      regime == "ma" ~ "Machine Annotation",
      TRUE ~ regime
    ),
    levels = c("Best", "Expert", "Subjective", "Machine Annotation")
  )
)

readr::write_csv(tidyK, file.path(out_dir, sprintf("cbm_tidy_%s.csv", opt$run_name)))
writeLines(runs$viab, con = file.path(out_dir, sprintf("manifest_%s.txt", opt$run_name)))

theme_white <- theme_minimal(base_size = 13) +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin      = margin(25, 25, 0, 0),
    legend.position        = "inside",
    legend.position.inside = c(0.98, 0.02),
    legend.justification   = c(1, 0),
    legend.background      = element_rect(fill = "white", colour = NA),
    axis.title.x = element_text(size = rel(1.5)),
    axis.title.y = element_text(size = rel(1.5)),
    legend.text  = element_text(size = rel(1.5))
  )

line_layer <- list(
  geom_line(linewidth = 0.8, alpha = 0.95),
  geom_point(size = 2, alpha = 0.95)
)

x_scale_budgetK <- function(breaks) scale_x_continuous(breaks = breaks, limits = c(0, max(breaks)), expand = expansion(mult = c(0, 0.04)))
x_scale_checks  <- function(x)      scale_x_continuous(limits = c(0, max(x, na.rm = TRUE)), expand = expansion(mult = c(0, 0.04)))
y_scale_pct_pos <- function()       scale_y_continuous(limits = c(0, 1), labels = label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.04)))
y_scale_pct_auto <- function(y) {
  rng <- range(y, na.rm = TRUE)
  scale_y_continuous(limits = c(min(0, rng[1]), max(0, rng[2])),
                     labels = label_percent(accuracy = 1),
                     expand = expansion(mult = c(0, 0.04)))
}
y_scale_num_pos <- function(ymax)   scale_y_continuous(limits = c(0, ymax), expand = expansion(mult = c(0, 0.04)))

breaksK <- sort(unique(tidyK$budget))
p_acc_K <- ggplot(tidyK, aes(x = budget, y = acc_post, color = regime_label)) +
  line_layer + x_scale_budgetK(breaksK) + y_scale_pct_pos() +
  labs(x = "Check Budget", y = "Accuracy", color = NULL) + theme_white
ggsave(file.path(out_dir, sprintf("CBM_acc_vs_budget_%s.pdf", opt$run_name)), p_acc_K, width = 8, height = 5, device = "pdf", bg = "white")

p_acc_vs45 <- ggplot(tidyK, aes(x = budget, y = acc_post, color = regime_label)) +
  line_layer + x_scale_budgetK(breaksK) +
  scale_y_continuous(
    limits = c(0.45, 1),
    labels = label_percent(accuracy = 1),
    expand = expansion(mult = c(0, 0.04))
  ) +
  labs(x = "Check Budget", y = "Accuracy", color = NULL) +
  theme_white +
  theme(
    legend.position        = "inside",
    legend.position.inside = c(0.98, 0.12),
    legend.justification   = c(1, 0),
    legend.background      = element_rect(fill = "transparent", colour = NA),
    legend.key             = element_rect(fill = "transparent", colour = NA)
  )
ggsave(file.path(out_dir, sprintf("CBM_acc_vs_budget_vs45_%s.pdf", opt$run_name)), p_acc_vs45, width = 8, height = 5, device = "pdf", bg = "white")

tk_checks <- tidyK %>% filter(!is.na(checks_per_case))
if (nrow(tk_checks) > 0) {
  p_acc_checks <- ggplot(tk_checks, aes(x = checks_per_case, y = acc_post, color = regime_label)) +
    line_layer + x_scale_checks(tk_checks$checks_per_case) + y_scale_pct_pos() +
    labs(x = "Check Budget", y = "Accuracy", color = NULL) + theme_white
  ggsave(file.path(out_dir, sprintf("CBM_acc_vs_checkbudget_%s.pdf", opt$run_name)), p_acc_checks, width = 8, height = 5, device = "pdf", bg = "white")
}

tk_gain <- tk_checks %>%
  group_by(regime) %>%
  mutate(gain_k0 = first(acc_post)) %>%
  ungroup() %>%
  mutate(gain = ifelse(!is.na(raw_gain_vs_k0), raw_gain_vs_k0, acc_post - gain_k0))
if (nrow(tk_gain) > 0) {
  p_gain <- ggplot(tk_gain, aes(x = checks_per_case, y = gain, color = regime_label)) +
    line_layer + x_scale_checks(tk_gain$checks_per_case) + y_scale_pct_auto(tk_gain$gain) +
    labs(x = "Check Budget", y = "Accuracy Gain", color = NULL) + theme_white
  ggsave(file.path(out_dir, sprintf("CBM_gain_vs_checkbudget_%s.pdf", opt$run_name)), p_gain, width = 8, height = 5, device = "pdf", bg = "white")
}

tk_delta <- tk_checks %>% mutate(delta = delta_vs_blackbox)
if (nrow(tk_delta) > 0 && any(!is.na(tk_delta$delta))) {
  p_delta <- ggplot(tk_delta, aes(x = checks_per_case, y = delta, color = regime_label)) +
    line_layer + x_scale_checks(tk_delta$checks_per_case) + y_scale_pct_auto(tk_delta$delta) +
    labs(x = "Check Budget", y = "Accuracy - DNN", color = NULL)+ theme_white
  ggsave(file.path(out_dir, sprintf("CBM_vsDNN_delta_vs_checkbudget_%s.pdf", opt$run_name)), p_delta, width = 8, height = 5, device = "pdf", bg = "white")
}
# --------- Concept error at k=0 (unchanged, soft; table uses its own hard validation) ---------
compute_concept_error_from_viab_k0 <- function(viab_path, regime) {
  if (is.na(viab_path)) return(NA_real_)
  df <- suppressMessages(readr::read_csv(viab_path, show_col_types = FALSE))
  if (!("budget" %in% names(df))) {
    if ("k" %in% names(df)) df <- dplyr::rename(df, budget = k) else return(NA_real_)
  }
  if ("concept_source" %in% names(df)) {
    want <- if (regime == "ma") "machine" else "detected"
    df <- df %>% dplyr::filter(concept_source == want)
  }
  if ("target_acc" %in% names(df)) df <- df %>% dplyr::filter(target_acc == "raw")
  k0 <- df %>% dplyr::filter(budget == 0)
  if (nrow(k0) == 0) return(NA_real_)
  nms <- names(k0)
  acc_col <- find_first_regex(nms, c("^pre_?concept_?acc", "concept_?acc(_?k?0)?$", "^concept_?accuracy(_?pre|_?k?0)?$"))
  if (!is.na(acc_col)) return(1 - agg_fun(k0[[acc_col]]))
  err_col <- find_first_regex(nms, c("^pre_?concept_?err(or)?", "concept_?err(or)?(_?k?0)?$", "^concept_?error(_?pre|_?k?0)?$"))
  if (!is.na(err_col)) return(agg_fun(k0[[err_col]]))
  num_col <- find_first_regex(nms, c("^n_?incorrect.*k?0?$", "^incorrect_?concepts(_?k?0)?$", "^n_?wrong_?concepts(_?k?0)?$"))
  den_col <- find_first_regex(nms, c("^n_?total_?concepts$", "^n_?concepts$", "^concepts_?total$"))
  if (!is.na(num_col) && !is.na(den_col)) return(sum(k0[[num_col]], na.rm = TRUE) / sum(k0[[den_col]], na.rm = TRUE))
  NA_real_
}

compute_concept_error_from_perconcept_k0 <- function(fp) {
  if (is.na(fp)) return(NA_real_)
  df <- suppressMessages(readr::read_csv(fp, show_col_types = FALSE))
  nms <- names(df)
  if ("is_correct" %in% nms) return(mean(1 - as.numeric(df$is_correct), na.rm = TRUE))
  if ("correct" %in% nms) return(mean(1 - as.numeric(df$correct), na.rm = TRUE))
  if (all(c("pred","truth") %in% nms)) return(mean(df$pred != df$truth, na.rm = TRUE))
  if (all(c("y_pred","y_true") %in% nms)) return(mean(df$y_pred != df$y_true, na.rm = TRUE))
  if (all(c("n_incorrect","n_total") %in% nms)) return(sum(df$n_incorrect, na.rm = TRUE) / sum(df$n_total, na.rm = TRUE))
  if (all(c("incorrect","total") %in% nms)) return(sum(df$incorrect, na.rm = TRUE) / sum(df$total, na.rm = TRUE))
  NA_real_
}

ce_viab <- purrr::pmap_dbl(list(runs$viab, runs$regime, seq_along(runs$viab)), function(v, r, i) compute_concept_error_from_viab_k0(v, r))
ce_pcon <- purrr::map_dbl(runs$per_concept_k0, compute_concept_error_from_perconcept_k0)
runs$concept_error_k0 <- ifelse(!is.na(ce_viab), ce_viab, ce_pcon)
# ---------------------------------------------------------------------------------------------

# ---------- Strict table path: validate + extract at k=table_budget from viability ------------
# Validate every regime loudly (this is independent of the 3x3 scan)
purrr::pwalk(list(runs$viab, runs$regime), \(v, r) validate_viability_for_table(v, r, opt$table_budget))

# ===== 3×3 TABLE (HumanAcc {70,80,100}% × Noise {0,20,30}%) — NO KEY FILTERS =====

# Re-scan runs for the table WITHOUT cbm_* gating
runs_table <- map_dfr(robot_roots, list_runs) %>%
  dplyr::filter(stringr::str_detect(rundir, "^(best|expert|subjective)_cbm_")) %>%
  dplyr::filter(!stringr::str_detect(rundir, "anchor")) %>%
  { if (tolower(opt$match_mode) == "exact")
      dplyr::filter(., rundir == opt$run_name)
    else
      dplyr::filter(., stringr::str_detect(rundir, stringr::fixed(opt$run_name))) } %>%
  dplyr::mutate(
    orig_regime = stringr::str_match(rundir, "^(best|expert|subjective)_cbm_")[,2],
    is_ma = orig_regime == "best" & stringr::str_detect(rundir, stringr::fixed("lfcbm")),
    regime = dplyr::if_else(is_ma, "ma", orig_regime)
  )

if (nrow(runs_table) == 0) stop("Table build: No runs matched run_name under robot_text.", call. = FALSE)

# Attach viability CSVs
runs_table <- runs_table %>%
  dplyr::mutate(viab = purrr::map_chr(path, pick_viability)) %>%
  dplyr::filter(!is.na(viab))

if (nrow(runs_table) == 0) stop("Table build: No viability CSVs found for matched runs.", call. = FALSE)

# Parse discrete factors from run names
parse_table_factors <- function(name) {
  h <- suppressWarnings(as.integer(stringr::str_match(name, "intervene(\\d+)")[,2]))
  n <- suppressWarnings(as.integer(stringr::str_match(name, "noise(\\d+)")[,2]))
  if (is.na(n) && grepl("^subjective_cbm_", name)) n <- 20L
  if (is.na(n)) n <- 0L
  # keep only allowed levels
  if (!(h %in% c(70L, 80L, 100L))) return(tibble(human_level = NA_integer_, noise_level = NA_integer_))
  if (!(n %in% c(0L, 20L, 30L)))   return(tibble(human_level = NA_integer_, noise_level = NA_integer_))
  tibble(human_level = h, noise_level = n)
}

runs_table <- runs_table %>%
  dplyr::bind_cols(purrr::map_dfr(runs_table$rundir, parse_table_factors)) %>%
  dplyr::filter(!is.na(human_level), !is.na(noise_level))

if (nrow(runs_table) == 0) stop("Table build: No runs have allowed intervene{70,80,100} × noise{0,20,30}.", call. = FALSE)

# Validate viability inputs loudly for each run at the requested table k
purrr::pwalk(
  list(runs_table$viab, runs_table$regime),
  \(v, r) validate_viability_for_table(v, r, opt$table_budget)
)

# Extract metrics strictly from viability at k=table_budget
extract_human_delta_from_viab <- function(viab_path, regime, table_k) {
  df <- suppressMessages(readr::read_csv(viab_path, show_col_types = FALSE))
  if (!("budget" %in% names(df))) {
    if ("k" %in% names(df)) df <- dplyr::rename(df, budget = k) else {
      stop("Missing budget/k in ", viab_path, ".", call. = FALSE)
    }
  }
  if ("concept_source" %in% names(df)) {
    want <- if (regime == "ma") "machine" else "detected"
    df <- df %>% dplyr::filter(concept_source == want)
    if (nrow(df) == 0) stop("concept_source filter removed all rows in ", viab_path, " for regime=", regime, ".", call. = FALSE)
  }
  if ("target_acc" %in% names(df)) {
    df <- df %>% dplyr::filter(target_acc == "raw")
    if (nrow(df) == 0) stop("No rows with target_acc=='raw' in ", viab_path, ".", call. = FALSE)
  }
  if (!("raw_gain_vs_k0" %in% names(df))) {
    stop("viability missing raw_gain_vs_k0: ", viab_path, ".", call. = FALSE)
  }
  krow <- df %>% dplyr::filter(budget == table_k)
  if (nrow(krow) == 0) stop("No budget==", table_k, " rows in ", viab_path, ".", call. = FALSE)

  esr_col <- pick_col(names(krow), c("edit_success_rate","human_acc","intervention_success_rate"))
  human <- if (!is.na(esr_col)) {
    agg_fun(krow[[esr_col]])
  } else {
    at_col  <- pick_col(names(krow), c("attempted_edits_total","attempted_edits"))
    cor_col <- pick_col(names(krow), c("corrected_edits_total","corrected_edits"))
    if (!is.na(at_col) && !is.na(cor_col) && sum(krow[[at_col]], na.rm = TRUE) > 0) {
      sum(krow[[cor_col]], na.rm = TRUE) / sum(krow[[at_col]], na.rm = TRUE)
    } else {
      fail_col <- pick_col(names(krow), c("failed_interventions_pct","failed_interventions_rate"))
      if (!is.na(fail_col)) 1 - agg_fun(krow[[fail_col]]) else stop(
        "Cannot derive human accuracy at k=", table_k, " from ", viab_path, ".",
        call. = FALSE
      )
    }
  }

  delta <- agg_fun(krow[["raw_gain_vs_k0"]])
  if (is.na(delta)) stop("raw_gain_vs_k0 at k=", table_k, " is NA in ", viab_path, ".", call. = FALSE)

  tibble(human_acc_k = human, delta_pp = 100 * delta)
}

metrics_tbl <- purrr::pmap_dfr(
  list(runs_table$viab, runs_table$regime, list(opt$table_budget)),
  extract_human_delta_from_viab
)

runs_table <- dplyr::bind_cols(runs_table, metrics_tbl)

# Build 3×3 grid
row_levels <- c("100%", "80%", "70%")  # top→bottom
col_levels <- c("0%", "20%", "30%")    # left→right

table_df <- runs_table %>%
  dplyr::mutate(
    human_str = paste0(human_level, "%"),
    noise_str = paste0(noise_level, "%")
  ) %>%
  dplyr::group_by(human_str, noise_str) %>%
  dplyr::summarise(delta_pp = agg_fun(delta_pp), .groups = "drop") %>%
  dplyr::mutate(cell = sprintf("%+.1f", delta_pp))

# Full grid; do NOT stop if missing—fill TeX with × and leave numeric as NA. Warn instead.
all_grid <- tidyr::expand_grid(
  human_str = factor(row_levels, levels = row_levels),
  noise_str  = factor(col_levels, levels = col_levels)
)

pivot_tex <- all_grid %>%
  dplyr::left_join(table_df %>% dplyr::select(human_str, noise_str, cell), by = c("human_str","noise_str")) %>%
  dplyr::mutate(cell = dplyr::coalesce(cell, "×")) %>%
  tidyr::pivot_wider(names_from = noise_str, values_from = cell) %>%
  dplyr::mutate(human_str = factor(human_str, levels = row_levels)) %>%
  dplyr::arrange(human_str)

pivot_num <- all_grid %>%
  dplyr::left_join(table_df %>% dplyr::select(human_str, noise_str, delta_pp), by = c("human_str","noise_str")) %>%
  tidyr::pivot_wider(names_from = noise_str, values_from = delta_pp) %>%
  dplyr::mutate(human_str = factor(human_str, levels = row_levels)) %>%
  dplyr::arrange(human_str)

# Warn about missing cells so it's still loud without killing the run
missing <- all_grid %>%
  dplyr::anti_join(table_df %>% dplyr::select(human_str, noise_str), by = c("human_str","noise_str"))
if (nrow(missing) > 0) {
  msg <- missing %>% dplyr::mutate(x = paste0("human=", human_str, ", noise=", noise_str)) %>% dplyr::pull(x) %>% paste(collapse=" | ")
  warning("3x3 table: missing cells filled with ×: ", msg)
}

# Write outputs
readr::write_csv(
  pivot_num,
  file.path(out_dir, sprintf("table_delta_acc_k%d_numeric_3x3_%s.csv", opt$table_budget, opt$run_name))
)

latex_escape <- function(x) gsub("%", "\\\\%", x, fixed = TRUE)
col_levels_tex <- latex_escape(col_levels)
align <- paste0("l|", paste(rep("c", length(col_levels_tex)), collapse = ""))

lines <- c(
  sprintf("\\begin{tabular}{%s}", align),
  paste(c("", col_levels_tex), collapse = " & "),
  "\\hline"
)
for (i in seq_len(nrow(pivot_tex))) {
  row_label <- as.character(pivot_tex$human_str[i])
  row_cells <- as.character(pivot_tex[i, col_levels, drop = FALSE] %>% as.vector() %>% unlist())
  row_cells <- latex_escape(row_cells)
  lines <- c(lines, paste(c(row_label, row_cells), collapse = " & "))
}
lines <- c(lines, "\\end{tabular}")
writeLines(
  lines,
  con = file.path(out_dir, sprintf("table_delta_acc_k%d_by_humanacc_x_noise_3x3_%s.tex", opt$table_budget, opt$run_name))
)
# ===== END 3×3 TABLE =====
