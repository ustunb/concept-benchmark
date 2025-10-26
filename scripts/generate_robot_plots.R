#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(stringr)
  library(tidyr)
  library(tools)
})

logi <- function(fmt, ...) cat(sprintf(paste0("[info] ", fmt, "\n"), ...))
loge <- function(fmt, ...) cat(sprintf(paste0("[error] ", fmt, "\n"), ...))

# ---------- helpers ----------

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- list()
  for (a in args) {
    if (grepl("^--", a)) {
      sp <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
      key <- sp[1]
      val <- ifelse(length(sp) >= 2, paste(sp[-1], collapse="="), "1")
      kv[[key]] <- val
    }
  }
  if (is.null(kv$repo_root)) kv$repo_root <- getwd()
  if (is.null(kv$results_root)) kv$results_root <- file.path(kv$repo_root, "results")
  if (is.null(kv$run_name) && is.null(kv$run_glob) && is.null(kv$run_prefix)) {
    stop("Provide one of: --run_name, --run_glob, or --run_prefix")
  }
  # optional: --expert_acc=0.8 or 80 (filters to only that expert line)
  if (!is.null(kv$expert_acc)) {
    acc <- suppressWarnings(as.numeric(kv$expert_acc))
    if (is.finite(acc)) {
      if (acc > 1) acc <- acc / 100
      kv$expert_acc <- acc
    } else {
      kv$expert_acc <- NULL
    }
  }
  if (is.null(kv$outdir)) kv$outdir <- file.path(kv$results_root, "robot_text")
  kv
}

ensure_dir <- function(p) { if (!dir.exists(p)) dir.create(p, recursive = TRUE, showWarnings = FALSE) }

find_run_dirs <- function(results_root, run_name=NULL, run_glob=NULL, run_prefix=NULL) {
  base <- file.path(results_root, "robot_text")
  if (!dir.exists(base)) return(character(0))
  dirs <- list.dirs(base, full.names = TRUE, recursive = FALSE)
  # drop any *_plots dirs
  dirs <- dirs[!grepl("_plots$", basename(dirs))]
  pick <- character(0)
  if (!is.null(run_name)) {
    cand <- file.path(base, run_name)
    if (dir.exists(cand)) pick <- c(pick, cand)
  }
  if (!is.null(run_glob)) {
    g <- Sys.glob(file.path(base, run_glob))
    g <- g[file.exists(g)]
    pick <- c(pick, g)
  }
  if (!is.null(run_prefix)) {
    dnames <- basename(dirs)
    rx <- paste0("^(cbm|lfcbm)_.+", run_prefix, "_cbm_(best|expert)_")
    keep <- grepl(rx, dnames)
    pick <- c(pick, dirs[keep])
  }
  unique(normalizePath(pick, mustWork = FALSE))
}

parse_run_attrs <- function(run_name) {
  # concept_source from prefix
  cs <- if (startsWith(run_name, "lfcbm_")) "machine" else if (startsWith(run_name, "cbm_")) "detected" else "unknown"
  # regime
  m <- regexpr("_cbm_(best|expert)_", run_name)
  regime <- ifelse(m > 0, gsub("^_cbm_|_$", "", regmatches(run_name, m)), NA_character_)
  # subkey between prefix and _cbm_
  m2 <- regexec("^(cbm|lfcbm)_(.+?)_cbm_", run_name)
  g <- regmatches(run_name, m2)[[1]]
  subkey <- if (length(g) >= 3) g[3] else NA_character_
  # human_acc from 'interveneXX' in name
  m3 <- regexec("intervene(\\d+)", run_name)
  g3 <- regmatches(run_name, m3)[[1]]
  acc_pct <- if (length(g3) >= 2) suppressWarnings(as.numeric(g3[2])) else NA_real_
  acc <- if (is.finite(acc_pct)) acc_pct / 100 else NA_real_
  list(concept_source = cs, regime = regime, subkey = subkey, expert_acc = acc)
}

read_viability <- function(run_dir) {
  # read only from <run>, not from <run>_plots
  cand <- list.files(run_dir, pattern = "^viability_robots_text_complete_seed.*_(detected|machine)\\.csv$", full.names = TRUE)
  if (!length(cand)) return(NULL)
  df <- suppressWarnings(readr::read_csv(cand[1], show_col_types = FALSE))
  required <- c("budget", "acc_cbm_pre", "acc_cbm_intv")
  if (!all(required %in% names(df))) return(NULL)
  df <- df %>% select(any_of(c("target_acc","budget","acc_cbm_pre","acc_cbm_intv","raw_gain_vs_k0","corrected_edits_total","attempted_edits_total")))
  df
}

plot_overlay <- function(A, subkey, cs_label, out_root) {
  if (!nrow(A)) return(invisible(NULL))
  # Build series labels: best and expert-<acc>
  A <- A %>% mutate(series = ifelse(regime == "best",
                                    "best",
                                    ifelse(is.finite(expert_acc), paste0("expert-", round(100*expert_acc)), "expert")))
  # one combined plot with all series
  p <- ggplot(A, aes(x = budget, y = acc_cbm_intv, color = series, linetype = series, shape = series)) +
    geom_line() + geom_point(size = 2) +
    scale_x_continuous(breaks = sort(unique(A$budget))) +
    ylim(0, 1) +
    labs(x = "Intervention budget (k)",
         y = "Post-intervention accuracy",
         title = paste0("Accuracy vs budget — ", subkey, " — ", cs_label)) +
    theme_bw()
  out_png = file.path(out_root, paste0("acc_vs_budget_", subkey, "_", gsub("[() ]","", cs_label), ".png"))
  ggsave(out_png, plot = p, width = 8.0, height = 4.5, dpi = 200)
}

main <- function() {
  args <- parse_args()
  repo_root <- normalizePath(args$repo_root, mustWork = TRUE)
  results_root <- normalizePath(args$results_root, mustWork = FALSE)
  out_root <- normalizePath(args$outdir, mustWork = FALSE)
  ensure_dir(out_root)

  runs <- find_run_dirs(results_root, run_name = args$run_name, run_glob = args$run_glob, run_prefix = args$run_prefix)
  if (!length(runs)) {
    loge("No runs found under %s/robot_text matching selector(s).", results_root)
    quit(status = 1)
  }
  readr::write_csv(data.frame(run_dir = runs), file.path(out_root, "RUNS_FOUND.csv"))
  logi("Found %d run(s).", length(runs))

  # Aggregate all
  rows <- list()
  for (rd in runs) {
    rn <- basename(rd)
    attrs <- parse_run_attrs(rn)
    v <- read_viability(rd)
    if (is.null(v)) next
    v$run <- rn
    v$subkey <- attrs$subkey
    v$regime <- attrs$regime
    v$concept_source <- attrs$concept_source
    v$expert_acc <- attrs$expert_acc
    rows[[length(rows)+1]] <- v
  }
  if (!length(rows)) {
    loge("No viability CSVs found in the selected runs.")
    quit(status = 1)
  }
  A <- dplyr::bind_rows(rows)

  # Optional: filter to one expert accuracy
  if (!is.null(args$expert_acc)) {
    A <- A %>% filter(regime == "expert" & is.finite(expert_acc) & abs(expert_acc - as.numeric(args$expert_acc)) < 1e-6)
    if (!nrow(A)) {
      loge("No expert runs match --expert_acc=%s", as.character(args$expert_acc))
      quit(status = 1)
    }
  }

  # Map concept source labels
  A$cs_label <- dplyr::case_when(
    A$concept_source == "detected" ~ "Automated Detection",
    A$concept_source == "machine"  ~ "Machine Concepts (LFCBM)",
    TRUE ~ "Unknown"
  )

  # Per-run quick products + per-subkey overlays
  # Per-run
  for (rd in unique(A$run)) {
    Ak <- A %>% filter(run == rd) %>% distinct(budget, acc_cbm_pre, acc_cbm_intv, .keep_all = TRUE)
    outdir <- file.path(out_root, paste0(rd, "_plots"))
    ensure_dir(outdir)
    readr::write_csv(Ak %>% select(budget, acc_cbm_pre, acc_cbm_intv), file.path(outdir, "acc_vs_budget.csv"))
    # simple per-run plot (post only)
    p <- ggplot(Ak, aes(x = budget, y = acc_cbm_intv)) +
      geom_line() + geom_point() +
      ylim(0, 1) +
      scale_x_continuous(breaks = sort(unique(Ak$budget))) +
      labs(x = "Intervention budget (k)", y = "Post-intervention accuracy", title = rd) +
      theme_bw()
    ggsave(file.path(outdir, "acc_vs_budget.png"), plot = p, width = 7.0, height = 4.0, dpi = 200)
  }

  # Combined overlays per subkey and concept-source label
  keys <- unique(A$subkey)
  for (k in keys) {
    Ak <- A %>% filter(subkey == k)
    # per-source overlays
    for (cs in unique(Ak$cs_label)) {
      G <- Ak %>% filter(cs_label == cs) %>% select(budget, acc_cbm_intv, regime, expert_acc, cs_label)
      plot_overlay(G, k, cs, out_root)
    }
    # combined overlay across sources
    plot_overlay_combined(Ak %>% select(budget, acc_cbm_intv, regime, expert_acc, cs_label), k, out_root)
  }



plot_overlay_combined <- function(A, subkey, out_root) {
  if (!nrow(A)) return(invisible(NULL))
  A <- A %>% mutate(
    series = ifelse(regime == "best",
                    "best",
                    ifelse(is.finite(expert_acc), paste0("expert-", round(100*expert_acc)), "expert"))
  )
  A <- A %>% mutate(label = paste0(cs_label, " — ", series))
  p <- ggplot(A, aes(x = budget, y = acc_cbm_intv, color = label, linetype = cs_label, shape = series)) +
    geom_line() + geom_point(size = 2) +
    scale_x_continuous(breaks = sort(unique(A$budget))) +
    ylim(0, 1) +
    labs(x = "Intervention budget (k)",
         y = "Post-intervention accuracy",
         title = paste0("Accuracy vs budget — ", subkey)) +
    theme_bw()
  out_png = file.path(out_root, paste0("acc_vs_budget_", subkey, "_combined.png"))
  ggsave(out_png, plot = p, width = 9.0, height = 4.8, dpi = 200)
}

readr::write_csv(A, file.path(out_root, "acc_vs_budget_all_runs.csv"))
  logi("[done] Wrote outputs to: %s", out_root)
}

if (identical(environment(), globalenv())) {
  tryCatch(main(), error = function(e) { loge("%s", e$message); quit(status = 1) })
}
