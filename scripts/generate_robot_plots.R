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
  cs <- if (startsWith(run_name, "lfcbm_")) "machine" else if (startsWith(run_name, "cbm_")) "detected" else "unknown"
  m <- regexpr("_cbm_(best|expert)_", run_name)
  regime <- ifelse(m > 0, gsub("^_cbm_|_$", "", regmatches(run_name, m)), NA_character_)
  m2 <- regexec("^(cbm|lfcbm)_(.+?)_cbm_", run_name)
  g <- regmatches(run_name, m2)[[1]]
  subkey <- if (length(g) >= 3) g[3] else NA_character_
  m3 <- regexec("intervene(\\d+)", run_name)
  g3 <- regmatches(run_name, m3)[[1]]
  acc_pct <- if (length(g3) >= 2) suppressWarnings(as.numeric(g3[2])) else NA_real_
  acc <- if (is.finite(acc_pct)) acc_pct / 100 else NA_real_
  m4 <- regexec("(?:noise|cn)(\\d+)", run_name, perl = TRUE)
  g4 <- regmatches(run_name, m4)[[1]]
  nz <- if (length(g4) >= 2) suppressWarnings(as.numeric(g4[2])) else NA_real_
  list(concept_source = cs, regime = regime, subkey = subkey, expert_acc = acc, noise_pct = nz)
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
  A <- A %>% mutate(
    series = dplyr::case_when(
      startsWith(run, "lfcbm_") ~ "Automated Detection",
      regime == "best" ~ "Perfect Annotation",
      regime == "expert" ~ "Expert Annotations",
      TRUE ~ "Other"
    ),
    series = factor(series, levels = c("Automated Detection","Expert Annotations","Perfect Annotation","Other"))
  )
  p <- ggplot(A, aes(x = budget, y = acc_cbm_intv, color = series)) +
    geom_line() + geom_point(size = 2) +
    scale_x_continuous(breaks = sort(unique(A$budget))) +
    ylim(0, 1) +
    labs(x = "Intervention budget (k)",
         y = "Post-intervention accuracy",
         title = paste0("Accuracy vs budget — ", subkey)) +
    theme_bw()
  out_png = file.path(out_root, paste0("acc_vs_budget_", subkey, ".png"))
  ggsave(out_png, plot = p, width = 8.0, height = 4.5, dpi = 200)
}

plot_overlay_combined <- function(A, subkey, out_root) {
  if (!nrow(A)) return(invisible(NULL))
  A <- A %>% mutate(
    series = dplyr::case_when(
      startsWith(run, "lfcbm_") ~ "Automated Detection",
      regime == "best" ~ "Perfect Annotation",
      regime == "expert" & is.finite(expert_acc) ~ paste0("Expert Annotations — ", round(100*expert_acc)),
      TRUE ~ "Other"
    )
  )
  expert_levels <- sort(unique(A$series[grepl("^Expert Annotations", A$series)]))
  A$series <- factor(A$series, levels = unique(c("Automated Detection", expert_levels, "Perfect Annotation", "Other")))

  min_pct <- suppressWarnings(min(100*A$acc_cbm_intv, na.rm = TRUE))
  y0_pct <- if (is.finite(min_pct)) { s <- floor(min_pct/5)*5; if (s >= min_pct) s <- s - 5; max(0, s) } else 0
  y0 <- y0_pct/100
  xmax <- suppressWarnings(max(A$budget, na.rm = TRUE)); if (!is.finite(xmax)) xmax <- 1

  p <- ggplot(A, aes(x = budget, y = acc_cbm_intv, color = series)) +
    geom_line() + geom_point(size = 2) +
    scale_x_continuous(breaks = sort(unique(A$budget)), limits = c(0, xmax), expand = c(0,0)) +
    scale_y_continuous(limits = c(y0, 1), breaks = seq(y0, 1, by = 0.05)) +
    labs(x = "Intervention budget (k)", y = "Post-intervention accuracy") +
    theme_bw() +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          plot.title = element_blank()) +
    guides(color = guide_legend(title = NULL))
  out_png <- file.path(out_root, paste0("acc_vs_budget_", subkey, "_combined.png"))
  ggsave(out_png, plot = p, width = 9.0, height = 4.8, dpi = 200)
}

plot_overlay_per_expert <- function(A, subkey, out_root) {
  exps <- sort(unique(A$expert_acc[A$regime == "expert" & is.finite(A$expert_acc)]))
  if (!length(exps)) return(invisible(NULL))
  for (ea in exps) {
    B <- A %>% filter(
      (regime == "expert" & is.finite(expert_acc) & abs(expert_acc - ea) < 1e-9) |
      (regime == "best") |
      (startsWith(run, "lfcbm_"))
    ) %>% mutate(
      series = dplyr::case_when(
        startsWith(run, "lfcbm_") ~ "Automated Detection",
        regime == "best" ~ "Perfect Annotation",
        regime == "expert" ~ paste0("Expert Annotations — ", round(100*ea)),
        TRUE ~ "Other"
      )
    )
    B$series <- factor(B$series, levels = c("Automated Detection",
                                            paste0("Expert Annotations — ", round(100*ea)),
                                            "Perfect Annotation"))

    min_pct <- suppressWarnings(min(100*B$acc_cbm_intv, na.rm = TRUE))
    y0_pct <- if (is.finite(min_pct)) { s <- floor(min_pct/5)*5; if (s >= min_pct) s <- s - 5; max(0, s) } else 0
    y0 <- y0_pct/100
    xmax <- suppressWarnings(max(B$budget, na.rm = TRUE)); if (!is.finite(xmax)) xmax <- 1

    p <- ggplot(B, aes(x = budget, y = acc_cbm_intv, color = series)) +
      geom_line() + geom_point(size = 2) +
      scale_x_continuous(breaks = sort(unique(B$budget)), limits = c(0, xmax), expand = c(0,0)) +
      scale_y_continuous(limits = c(y0, 1), breaks = seq(y0, 1, by = 0.05)) +
      labs(x = "Intervention budget (k)", y = "Post-intervention accuracy") +
      theme_bw() +
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            plot.title = element_blank()) +
      guides(color = guide_legend(title = NULL))
    out_png <- file.path(out_root, paste0("acc_vs_budget_", subkey, "_expert_", round(100*ea), ".png"))
    ggsave(out_png, plot = p, width = 9.0, height = 4.8, dpi = 200)
  }
}

plot_overlay_combined_variants <- function(A, subkey, out_root) {
  bs <- sort(unique(A$budget))
  extras <- setdiff(bs, c(0,1))
  if (!length(extras)) return(invisible(NULL))
  for (b in extras) {
    S <- A %>% filter(budget %in% c(0,1,b))
    if (!nrow(S)) next
    S <- S %>% mutate(
      series = dplyr::case_when(
        startsWith(run, "lfcbm_") ~ "Automated Detection",
        regime == "best" ~ "Perfect Annotation",
        regime == "expert" & is.finite(expert_acc) ~ paste0("Expert Annotations — ", round(100*expert_acc)),
        TRUE ~ "Other"
      )
    )
    expert_levels <- sort(unique(S$series[grepl("^Expert Annotations", S$series)]))
    S$series <- factor(S$series, levels = unique(c("Automated Detection", expert_levels, "Perfect Annotation", "Other")))

    min_pct <- suppressWarnings(min(100*S$acc_cbm_intv, na.rm = TRUE))
    y0_pct <- if (is.finite(min_pct)) { s <- floor(min_pct/5)*5; if (s >= min_pct) s <- s - 5; max(0, s) } else 0
    y0 <- y0_pct/100
    xmax <- suppressWarnings(max(S$budget, na.rm = TRUE)); if (!is.finite(xmax)) xmax <- 1

    p <- ggplot(S, aes(x = budget, y = acc_cbm_intv, color = series)) +
      geom_line() + geom_point(size = 2) +
      scale_x_continuous(breaks = sort(unique(S$budget)), limits = c(0, xmax), expand = c(0,0)) +
      scale_y_continuous(limits = c(y0, 1), breaks = seq(y0, 1, by = 0.05)) +
      labs(x = "Intervention budget (k)", y = "Post-intervention accuracy") +
      theme_bw() +
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            plot.title = element_blank()) +
      guides(color = guide_legend(title = NULL))
    out_png <- file.path(out_root, paste0("acc_vs_budget_", subkey, "_combined_budgets_0_1_", b, ".png"))
    ggsave(out_png, plot = p, width = 9.0, height = 4.8, dpi = 200)
  }
}

plot_overlay_per_expert_variants <- function(A, subkey, out_root) {
  exps <- sort(unique(A$expert_acc[A$regime == "expert" & is.finite(A$expert_acc)]))
  if (!length(exps)) return(invisible(NULL))
  for (ea in exps) {
    B <- A %>% filter(
      (regime == "expert" & is.finite(expert_acc) & abs(expert_acc - ea) < 1e-9) |
      (regime == "best") |
      (startsWith(run, "lfcbm_"))
    ) %>% mutate(
      series = dplyr::case_when(
        startsWith(run, "lfcbm_") ~ "Automated Detection",
        regime == "best" ~ "Perfect Annotation",
        regime == "expert" ~ paste0("Expert Annotations — ", round(100*ea)),
        TRUE ~ "Other"
      )
    )
    bs <- sort(unique(B$budget))
    extras <- setdiff(bs, c(0,1))
    if (!length(extras)) next
    for (b in extras) {
      S <- B %>% filter(budget %in% c(0,1,b))
      if (!nrow(S)) next

      min_pct <- suppressWarnings(min(100*S$acc_cbm_intv, na.rm = TRUE))
      y0_pct <- if (is.finite(min_pct)) { s <- floor(min_pct/5)*5; if (s >= min_pct) s <- s - 5; max(0, s) } else 0
      y0 <- y0_pct/100
      xmax <- suppressWarnings(max(S$budget, na.rm = TRUE)); if (!is.finite(xmax)) xmax <- 1

      p <- ggplot(S, aes(x = budget, y = acc_cbm_intv, color = series)) +
        geom_line() + geom_point(size = 2) +
        scale_x_continuous(breaks = sort(unique(S$budget)), limits = c(0, xmax), expand = c(0,0)) +
        scale_y_continuous(limits = c(y0, 1), breaks = seq(y0, 1, by = 0.05)) +
        labs(x = "Intervention budget (k)", y = "Post-intervention accuracy") +
        theme_bw() +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
              plot.title = element_blank()) +
        guides(color = guide_legend(title = NULL))
      out_png <- file.path(out_root, paste0("acc_vs_budget_", subkey, "_expert_", round(100*ea), "_budgets_0_1_", b, ".png"))
      ggsave(out_png, plot = p, width = 9.0, height = 4.8, dpi = 200)
    }
  }
}


write_gain_table <- function(A, out_root) {
  G <- A %>% dplyr::filter(budget == 1)
  if (!nrow(G)) return(invisible(NULL))

  if ("noise_pct" %in% names(G)) {
    G$noise_col <- G$noise_pct
  } else if ("concept_noise" %in% names(G)) {
    G$noise_col <- suppressWarnings(round(100*as.numeric(G$concept_noise)))
  } else {
    G$noise_col <- NA_real_
  }

  if ("raw_gain_vs_k0" %in% names(G)) {
    G$gain <- G$raw_gain_vs_k0
  } else if (all(c("acc_cbm_intv","acc_cbm_pre") %in% names(G))) {
    G$gain <- G$acc_cbm_intv - G$acc_cbm_pre
  } else {
    return(invisible(NULL))
  }

  G$expert_label <- ifelse(is.finite(G$expert_acc), round(100*G$expert_acc), NA_real_)
  W <- G %>% dplyr::group_by(expert_label, noise_col) %>% dplyr::summarise(gain = mean(gain, na.rm = TRUE), .groups = "drop") %>% dplyr::arrange(dplyr::desc(expert_label), noise_col)
  if (!nrow(W)) return(invisible(NULL))

  W_wide <- tidyr::pivot_wider(W %>% dplyr::mutate(gain_pp = round(100*gain, 1)),
                               names_from = noise_col, values_from = gain_pp, names_sort = TRUE)
  readr::write_csv(W_wide, file.path(out_root, "accuracy_gain_table.csv"))

  noise_cols <- setdiff(names(W_wide), "expert_label")
  if (length(noise_cols) == 0) {
    lines <- c(
      "\\begin{wraptable}[13]{R}{0.45\\linewidth}",
      "    \\scriptsize",
      "    \\centering",
      "    \\vspace{-1.2em}",
      "    \\begin{tabular}{l|c}",
      "      & Gain (pp) \\\\",
      "        \\cmidrule(lr){2-2}",
      "        Expert Acc. & k=1 \\\\",
      "        \\hline"
    )
    for (i in seq_len(nrow(W_wide))) {
      r <- W_wide[i, ]
      val <- r[[2]]
      val_fmt <- ifelse(is.na(val), "$\\times$", paste0("+", as.character(val)))
      lines <- c(lines, paste0("        ", r$expert_label, "\\%  & ", val_fmt, " \\\\"))
    }
    lines <- c(lines,
      "    \\end{tabular}",
      "    \\caption{$Gain_{acc}$ from a single targeted concept intervention ($k=1$).}",
      "    \\label{tab:accuracy_gain_table}",
      "\\end{wraptable}"
    )
  } else {
    col_spec <- paste(rep("c", length(noise_cols)), collapse = "")
    hdr_noise <- paste0(noise_cols, "\\%")
    cmid_hi <- length(noise_cols) + 1
    lines <- c(
      "\\begin{wraptable}[13]{R}{0.45\\linewidth}",
      "    \\scriptsize",
      "    \\centering",
      "    \\vspace{-1.2em}",
      paste0("    \\begin{tabular}{l|", col_spec, "}"),
      paste0("      & \\multicolumn{", length(noise_cols), "}{c}{Concept Noise (\\%)} \\\\"),
      paste0("        \\cmidrule(lr){2-", cmid_hi, "}"),
      paste0("        Expert Acc. & ", paste(hdr_noise, collapse = " & "), " \\\\"),
      "        \\hline"
    )
    for (i in seq_len(nrow(W_wide))) {
      r <- W_wide[i, ]
      vals <- r[1, noise_cols, drop = TRUE]
      vals_fmt <- ifelse(is.na(vals), "$\\times$", paste0("+", as.character(vals)))
      lines <- c(lines, paste0("        ", r$expert_label, "\\%  & ", paste(vals_fmt, collapse = " & "), " \\\\"))
    }
    lines <- c(lines,
      "    \\end{tabular}",
      "    \\caption{$Gain_{acc}$ from a single targeted concept intervention ($k=1$). Rows represent the human expert's intervention accuracy, while columns show the percentage of missing concept annotations.}",
      "    \\label{tab:missingness_deltas}",
      "\\end{wraptable}"
    )
  }
  writeLines(lines, con = file.path(out_root, "accuracy_gain_table.tex"))
}

main <- function() {
  args <- parse_args()
  repo_root <- normalizePath(args$repo_root, mustWork = TRUE)
  results_root <- normalizePath(args$results_root, mustWork = FALSE)

  runs <- find_run_dirs(results_root, run_name = args$run_name, run_glob = args$run_glob, run_prefix = args$run_prefix)
  if (!length(runs)) {
    loge("No runs found under %s/robot_text matching selector(s).", results_root)
    quit(status = 1)
  }

  default_out <- file.path(results_root, "robot_text")
  req_out <- normalizePath(args$outdir, mustWork = FALSE)
  if (length(runs) == 1 && identical(req_out, normalizePath(default_out, mustWork = FALSE))) {
    out_root <- file.path(runs[[1]], "_plots")
  } else if (length(runs) > 1 && identical(req_out, normalizePath(default_out, mustWork = FALSE))) {
    out_root <- file.path(results_root, "robot_text", "_plots")
  } else {
    out_root <- req_out
  }
  ensure_dir(out_root)

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
    v$noise_pct <- attrs$noise_pct
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
  A$cs_label <- ifelse(startsWith(A$run, "lfcbm_"), "Automated Detection", NA_character_)

    # Only combined overlays per subkey + per-expert singletons + budget-variants
  keys <- unique(A$subkey)
  for (k in keys) {
    Ak <- A %>% filter(subkey == k)
    D <- Ak %>% select(run, budget, acc_cbm_intv, regime, expert_acc)
    plot_overlay_combined(D, k, out_root)
    plot_overlay_per_expert(D, k, out_root)
    plot_overlay_combined_variants(D, k, out_root)
    plot_overlay_per_expert_variants(D, k, out_root)
  }

  write_gain_table(A, out_root)
  readr::write_csv(A, file.path(out_root, "acc_vs_budget_all_runs.csv"))
  logi("[done] Wrote outputs to: %s", out_root)
}

if (identical(environment(), globalenv())) {
  tryCatch(main(), error = function(e) { loge("%s", e$message); quit(status = 1) })
}
