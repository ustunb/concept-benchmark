#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(stringr)
})

# ---------------- CLI ----------------
option_list <- list(
  make_option("--run_name", type="character", help="Exact run folder name OR substring (see --match_mode)"),
  make_option("--results_dir", type="character", default=NULL,
              help="Defaults to ../results relative to this script"),
  make_option("--out_subdir", type="character", default="demo_images",
              help="Subdir under results for outputs (default: demo_images)"),
  make_option("--match_mode", type="character", default="exact",
              help="exact | substring (default: exact)"),
  make_option("--expected_budgets", type="character", default="0,1,2,5,10",
              help="Comma list of K values to keep (default: 0,1,2,5,10)"),
  make_option("--target_acc", type="character", default="raw",
              help="Filter rows where target_acc == this (if column exists). Default: raw"),
  make_option("--concept_source", type="character", default="detected",
              help="Filter rows where concept_source == this (if column exists). Default: detected"),
  make_option("--agg", type="character", default="mean",
              help="Aggregation for duplicate K rows: mean | median (default: mean)")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$run_name) || nchar(opt$run_name) == 0)
  stop("run_name is required", call. = FALSE)

# ---------------- Paths ----------------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)])
script_dir <- if (length(script_path) == 0) getwd() else dirname(script_path)

results_dir <- if (!is.null(opt$results_dir)) {
  normalizePath(opt$results_dir, mustWork = TRUE)
} else {
  normalizePath(file.path(script_dir, "..", "results"), mustWork = TRUE)
}
out_dir <- file.path(results_dir, opt$out_subdir, opt$run_name)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("[make_viability_plots] results_dir = ", results_dir)
message("[make_viability_plots] run_name    = ", opt$run_name)
message("[make_viability_plots] match_mode  = ", opt$match_mode)
message("[make_viability_plots] out_dir     = ", out_dir)

exp_budgets <- str_split(opt$expected_budgets, ",", simplify = TRUE) %>% as.integer()
agg_fun <- if (tolower(opt$agg) == "median") \(x) median(x, na.rm = TRUE) else \(x) mean(x, na.rm = TRUE)

# ---------------- Discover viability CSVs ----------------
viab_pat <- "viability_robots_text_complete_.*_detected\\.csv$"
all_viabs <- list.files(results_dir, pattern = viab_pat, recursive = TRUE, full.names = TRUE)
if (length(all_viabs) == 0L) stop("No viability CSVs found in results_dir")

df_files <- tibble(
  path   = all_viabs,
  rundir = basename(dirname(all_viabs))
) %>%
  filter(!str_detect(rundir, "_anchor_")) %>%                              # drop anchor runs
  filter(str_detect(rundir, "^(best|expert|subjective)_(cbm|cs)_"))        # must be a proper run dir

if (tolower(opt$match_mode) == "exact") {
  df_files <- df_files %>% filter(rundir == opt$run_name)
} else {
  df_files <- df_files %>% filter(str_detect(rundir, fixed(opt$run_name)))
}

if (nrow(df_files) == 0L)
  stop("No matching run folders for run_name under results_dir with that match_mode")

# Parse regime/system from folder name, e.g., best_cbm_<...>
tags <- str_match(df_files$rundir, "^(best|expert|subjective)_(cbm|cs)_")
df_files <- df_files %>%
  mutate(regime = tags[,2], system = toupper(tags[,3]))

# Persist manifest/provenance
writeLines(df_files$path, con = file.path(out_dir, sprintf("manifest_%s.txt", opt$run_name)))
readr::write_csv(df_files, file.path(out_dir, sprintf("provenance_%s.csv", opt$run_name)))

# ---------------- Column pickers ----------------
pick_col <- function(nms, cands) {
  for (c in cands) if (c %in% nms) return(c)
  NA_character_
}
pick_post_acc <- function(nms, system) {
  if (system == "CBM") pick_col(nms, c("acc_cbm_intv","acc_cbm_post","acc_post","sel_acc_post"))
  else                 pick_col(nms, c("sel_acc_post","acc_cs_intv","acc_cs_post","acc_post","acc_cbm_intv"))
}
pick_pre_acc  <- function(nms) pick_col(nms, c("acc_cbm_pre","acc_cs_pre","acc_pre","sel_acc_pre"))

# ---------------- Read + tidy one file ----------------
read_viability <- function(fp, regime, system) {
  df0 <- suppressMessages(readr::read_csv(fp, show_col_types = FALSE))
  nms <- names(df0)
  if (!("budget" %in% nms)) {
    if ("k" %in% nms) df0 <- dplyr::rename(df0, budget = k) else stop("Missing budget column in: ", fp)
  }
  # Optional filters to avoid mixing rows
  if ("concept_source" %in% names(df0))
    df0 <- df0 %>% filter(.data$concept_source == opt$concept_source)
  if ("target_acc" %in% names(df0))
    df0 <- df0 %>% filter(.data$target_acc == opt$target_acc)

  post_col <- pick_post_acc(names(df0), system)
  pre_col  <- pick_pre_acc(names(df0))
  if (is.na(post_col)) stop("No post-accuracy column in: ", fp)

  # Keep only expected budgets; warn if missing
  present <- sort(intersect(df0$budget %>% as.integer(), exp_budgets))
  if (length(present) == 0L) {
    message("[skip] No expected budgets in: ", fp)
    return(tibble())
  }
  if (!setequal(present, exp_budgets))
    message("[warn] Missing budgets in ", basename(dirname(fp)), " -> have {", paste(present, collapse=","), "} expected {", paste(exp_budgets, collapse=","), "}")

  df1 <- df0 %>% filter(budget %in% present)

  # Aggregate duplicate rows per K (folds/seeds) with chosen agg
  safe_pull <- function(d, col) if (col %in% names(d)) d[[col]] else NA_real_
  df_agg <- df1 %>%
    group_by(budget) %>%
    summarize(
      acc_pre  = agg_fun(safe_pull(cur_data(), pre_col)),
      acc_post = agg_fun(safe_pull(cur_data(), post_col)),
      coverage_automated          = agg_fun(safe_pull(cur_data(), "coverage_automated")),
      coverage_after_confirmation = agg_fun(safe_pull(cur_data(), "coverage_after_confirmation")),
      avg_edits_per_case          = agg_fun(safe_pull(cur_data(), "avg_edits_per_case")),
      failed_interventions_pct    = agg_fun(safe_pull(cur_data(), "failed_interventions_pct")),
      concept_checks              = agg_fun(safe_pull(cur_data(), "concept_checks")),
      interventions_total         = agg_fun(safe_pull(cur_data(), "interventions_total")),
      .groups = "drop"
    ) %>%
    mutate(system = system, regime = regime) %>%
    relocate(system, regime, budget)

  df_agg
}

tidy <- pmap_dfr(
  list(df_files$path, df_files$regime, df_files$system),
  \(p, r, s) read_viability(p, r, s)
) %>% arrange(system, regime, budget)

if (nrow(tidy) == 0L) stop("No tidy rows after filtering; check --match_mode, --target_acc, --concept_source")

# Baseline at K=0 (fallback to min K per regime if 0 missing)
baseline <- tidy %>%
  group_by(system, regime) %>%
  summarize(
    baseline = {
      k0 <- acc_post[budget == 0]
      if (length(k0) > 0) k0[1] else acc_post[order(budget)][1]
    },
    .groups = "drop"
  )

tidy <- tidy %>%
  left_join(baseline, by = c("system","regime")) %>%
  mutate(acc_gain_vs_k0 = acc_post - baseline)

# Save tidy tables for inspection
readr::write_csv(tidy %>% filter(system == "CBM"), file.path(out_dir, sprintf("cbm_tidy_%s.csv", opt$run_name)))
readr::write_csv(tidy %>% filter(system == "CS"),  file.path(out_dir, sprintf("cs_tidy_%s.csv",  opt$run_name)))

# ---------------- Plot helper (WHITE, PDF) ----------------
theme_white_bg <- theme_minimal(base_size = 13) +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

plot_lines_pdf <- function(df, ycol, title, fname, ylab) {
  df <- df %>% filter(!is.na(.data[[ycol]]))
  if (nrow(df) == 0L) { message("[skip] ", fname, " (no data)"); return(invisible(NULL)) }
  breaks <- intersect(exp_budgets, sort(unique(df$budget)))
  p <- ggplot(df, aes(x = budget, y = .data[[ycol]], color = regime)) +
    geom_line() + geom_point() +
    scale_x_continuous(breaks = breaks) +
    labs(title = title, x = "Budget K", y = ylab, color = NULL) +
    theme_white_bg
  rng <- range(df[[ycol]], na.rm = TRUE)
  if (all(rng >= 0 & rng <= 1)) p <- p + coord_cartesian(ylim = c(0, 1))
  ggsave(file.path(out_dir, fname), p, width = 8, height = 5, dpi = 160, device = "pdf", bg = "white")
  message("[saved] ", file.path(out_dir, fname))
}

cbm <- tidy %>% filter(system == "CBM")
cs  <- tidy %>% filter(system == "CS")

# 1) Accuracy vs K (post)
plot_lines_pdf(cbm, "acc_post", sprintf("CBM — Accuracy vs Budget K (%s)", opt$run_name),
               sprintf("CBM_acc_vs_budget_%s.pdf", opt$run_name), "Post-intervention accuracy")
plot_lines_pdf(cs,  "acc_post", sprintf("CS — Accuracy vs Budget K (%s)", opt$run_name),
               sprintf("CS_acc_vs_budget_%s.pdf", opt$run_name),  "Post-intervention accuracy")

# 2) Accuracy gain vs K
plot_lines_pdf(cbm, "acc_gain_vs_k0", sprintf("CBM — Accuracy Gain vs Budget K (%s)", opt$run_name),
               sprintf("CBM_gain_vs_budget_%s.pdf", opt$run_name), "Accuracy gain (post − K=0)")
plot_lines_pdf(cs,  "acc_gain_vs_k0", sprintf("CS — Accuracy Gain vs Budget K (%s)", opt$run_name),
               sprintf("CS_gain_vs_budget_%s.pdf", opt$run_name),  "Accuracy gain (post − K=0)")

# 3) Coverage after confirmation vs K
plot_lines_pdf(cbm, "coverage_after_confirmation", sprintf("CBM — Coverage After Confirmation vs Budget K (%s)", opt$run_name),
               sprintf("CBM_coverage_after_vs_budget_%s.pdf", opt$run_name), "Coverage after confirmation")
plot_lines_pdf(cs,  "coverage_after_confirmation", sprintf("CS — Coverage After Confirmation vs Budget K (%s)", opt$run_name),
               sprintf("CS_coverage_after_vs_budget_%s.pdf", opt$run_name),  "Coverage after confirmation")

# 4) Automated coverage vs K
plot_lines_pdf(cbm, "coverage_automated", sprintf("CBM — Automated Coverage vs Budget K (%s)", opt$run_name),
               sprintf("CBM_coverage_automated_vs_budget_%s.pdf", opt$run_name), "Coverage (automated)")
plot_lines_pdf(cs,  "coverage_automated", sprintf("CS — Automated Coverage vs Budget K (%s)", opt$run_name),
               sprintf("CS_coverage_automated_vs_budget_%s.pdf", opt$run_name),  "Coverage (automated)")

# 5) Avg edits per case vs K
plot_lines_pdf(cbm, "avg_edits_per_case", sprintf("CBM — Avg Edits per Case vs Budget K (%s)", opt$run_name),
               sprintf("CBM_avg_edits_per_case_vs_budget_%s.pdf", opt$run_name), "Avg edits per case")
plot_lines_pdf(cs,  "avg_edits_per_case", sprintf("CS — Avg Edits per Case vs Budget K (%s)", opt$run_name),
               sprintf("CS_avg_edits_per_case_vs_budget_%s.pdf", opt$run_name),  "Avg edits per case")

# 6) Failed interventions (%) vs K
plot_lines_pdf(cbm, "failed_interventions_pct", sprintf("CBM — Failed Interventions %% vs Budget K (%s)", opt$run_name),
               sprintf("CBM_failed_interventions_pct_vs_budget_%s.pdf", opt$run_name), "Failed interventions (%)")
plot_lines_pdf(cs,  "failed_interventions_pct", sprintf("CS — Failed Interventions %% vs Budget K (%s)", opt$run_name),
               sprintf("CS_failed_interventions_pct_vs_budget_%s.pdf", opt$run_name),  "Failed interventions (%)")

message("[make_viability_plots] done")
