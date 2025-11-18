#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(dplyr)
  library(stringr)
  library(tools)
})

logi <- function(fmt, ...) cat(sprintf(paste0("[info] ", fmt, "\n"), ...))
loge <- function(fmt, ...) cat(sprintf(paste0("[error] ", fmt, "\n"), ...))

# ----- theme knobs -----
AXIS_TITLE_X_SIZE  <- 60  # numeric or rel; NULL uses default
AXIS_TITLE_Y_SIZE  <- 60
AXIS_TEXT_X_SIZE   <- 60  # tick labels on x-axis
AXIS_TEXT_Y_SIZE   <- 60  # tick labels on y-axis
LEGEND_TEXT_SIZE   <- 60
LEGEND_TITLE_SIZE  <- NULL
CAPTION_TEXT_SIZE  <- NULL
LEGEND_SHOW        <- TRUE  # TRUE = show legend, FALSE = hide legend

# Hard-coded subjective rates to KEEP (strings after "rate")
# e.g. rate10 -> "10", rate0.2 -> "0.2"
SUBJECTIVE_RATES_KEEP <- c("20")

# ----- arg parsing -----

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- list()
  i <- 1L
  while (i <= length(args)) {
    a <- args[i]
    if (grepl("^--", a)) {
      body <- sub("^--", "", a)
      if (grepl("=", body, fixed = TRUE)) {
        sp  <- strsplit(body, "=", fixed = TRUE)[[1]]
        key <- sp[1]
        val <- paste(sp[-1], collapse = "=")
      } else {
        key <- body
        if (i + 1L <= length(args) && !grepl("^--", args[i + 1L])) {
          val <- args[i + 1L]
          i   <- i + 1L
        } else {
          val <- "1"
        }
      }
      kv[[key]] <- val
    }
    i <- i + 1L
  }

  if (is.null(kv$repo_root)) {
    kv$repo_root <- getwd()
    logi("No --repo_root given; assuming repo_root = %s", kv$repo_root)
  }

  if (is.null(kv$results_root)) {
    # Default: results under results/robots/
    kv$results_root <- file.path(kv$repo_root, "results", "robots")
    logi("No --results_root given; assuming results_root = %s", kv$results_root)
  }

  if (is.null(kv$run_name)) {
    kv$run_name <- "scbm_run_1014"
    logi("No --run_name given; assuming run_name = %s", kv$run_name)
  }

  if (is.null(kv$outdir)) {
    kv$outdir <- file.path(kv$results_root, paste0(kv$run_name, "_plots"))
    logi("No --outdir given; assuming outdir = %s", kv$outdir)
  }

  kv
}

# ----- theme -----

build_base_theme <- function() {
  xt_size <- if (is.null(AXIS_TITLE_X_SIZE)) ggplot2::rel(1.5) else AXIS_TITLE_X_SIZE
  yt_size <- if (is.null(AXIS_TITLE_Y_SIZE)) ggplot2::rel(1.5) else AXIS_TITLE_Y_SIZE

  axis_text_x_size <- if (is.null(AXIS_TEXT_X_SIZE)) 15 else AXIS_TEXT_X_SIZE
  axis_text_y_size <- if (is.null(AXIS_TEXT_Y_SIZE)) 15 else AXIS_TEXT_Y_SIZE

  # Legend text shrunk by factor 0.8
  legend_text_size  <- if (is.null(LEGEND_TEXT_SIZE)) ggplot2::rel(1.5) else LEGEND_TEXT_SIZE * 0.8
  legend_title_size <- if (is.null(LEGEND_TITLE_SIZE)) ggplot2::rel(1.0) else LEGEND_TITLE_SIZE
  caption_size      <- if (is.null(CAPTION_TEXT_SIZE)) ggplot2::rel(1.0) else CAPTION_TEXT_SIZE

   theme_bw() +
    theme(
      # horizontal gridlines only
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "grey85", size = 0.3),
      panel.grid.minor.y = element_line(color = "grey92", size = 0.2),

      panel.border = element_blank(),
      axis.line.x  = element_line(color = "black", size = 1.0),
      axis.line.y  = element_line(color = "black", size = 1.0),

      plot.title   = element_blank(),

      # legend at bottom instead of inside the plot
      legend.position   = if (LEGEND_SHOW) "bottom" else "none",
      legend.box        = "horizontal",
      legend.box.just   = "left",
      legend.background = element_rect(fill = "white", color = NA),
      legend.key        = element_blank(),
      legend.text       = element_text(size = legend_text_size),
      legend.title      = element_blank(),  # no legend title
      plot.caption      = element_text(size = caption_size),

      axis.title.x = element_text(size = xt_size),
      axis.title.y = element_text(size = yt_size),
      axis.text.x  = element_text(size = axis_text_x_size),
      axis.text.y  = element_text(size = axis_text_y_size)
    )
}

.base_theme <- build_base_theme()

# ----- helpers for runs & setups -----

find_run_dirs <- function(results_root, run_name) {
  pattern <- file.path(results_root, paste0(run_name, "*"))
  logi("Searching for run dirs with pattern: %s", pattern)

  if (!dir.exists(results_root)) {
    loge("results_root does NOT exist: %s", results_root)
  } else {
    logi("results_root exists. Top-level entries: %s",
         paste(list.files(results_root), collapse = ", "))
  }

  cand <- Sys.glob(pattern)
  if (!length(cand)) {
    loge("No paths matched pattern: %s", pattern)
    return(character(0))
  }

  info <- file.info(cand)
  dirs <- cand[!is.na(info$isdir) & info$isdir]
  if (!length(dirs)) {
    loge("Matched paths but none were directories: %s",
         paste(cand, collapse = ", "))
  } else {
    logi("Found %d candidate run dirs: %s",
         length(dirs), paste(basename(dirs), collapse = ", "))
  }
  dirs
}

extract_setup <- function(run_dir) {
  bn <- basename(run_dir)
  m  <- regexec("__regime-([^/]+)$", bn)
  g  <- regmatches(bn, m)[[1]]
  if (length(g) >= 2) {
    g[2]
  } else {
    loge("Could not parse setup from directory name: %s (expected '__regime-<setup>')", bn)
    NA_character_
  }
}

# ----- metrics reading -----

read_metrics_single <- function(run_dir, setup, noise_level = NA_character_) {
  logi("Looking for metrics_*.json/jsonl in %s", run_dir)
  files <- list.files(
    run_dir,
    pattern   = "^metrics_.*\\.json(l)?$",
    full.names = TRUE,
    recursive  = FALSE
  )
  if (!length(files)) {
    loge("No metrics_*.json/jsonl file found in %s", run_dir)
    return(NULL)
  }

  path <- files[[1]]
  logi("Reading metrics from %s", path)

  metrics <- tryCatch(
    jsonlite::fromJSON(path, simplifyVector = FALSE),
    error = function(e) {
      loge("Failed to read JSON in %s: %s", path, e$message)
      NULL
    }
  )
  if (is.null(metrics) || is.null(metrics$interventions)) {
    loge("No 'interventions' block in %s", path)
    return(NULL)
  }

  ints <- metrics$interventions
  if (!length(ints)) {
    loge("Empty 'interventions' in %s", path)
    return(NULL)
  }

  rows <- lapply(names(ints), function(name) {
    obj  <- ints[[name]]
    step <- suppressWarnings(as.integer(sub("^top_(\\d+).*", "\\1", name)))
    if (!is.finite(step)) step <- NA_integer_
    gain <- obj$accuracy_gain
    if (is.null(gain)) gain <- NA_real_

    data.frame(
      run_dir          = run_dir,
      setup            = setup,
      noise_level      = noise_level,
      intervention_key = name,
      step             = step,
      gain             = as.numeric(gain),
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(rows)
}

load_all_metrics <- function(run_setups) {
  rows <- list()

  for (i in seq_len(nrow(run_setups))) {
    rd <- run_setups$run_dir[i]
    st <- run_setups$setup[i]
    logi("Processing run_dir=%s setup=%s", rd, st)

    if (st == "subjective") {
      rate_dirs <- list.dirs(rd, full.names = TRUE, recursive = FALSE)
      rate_dirs <- rate_dirs[basename(rate_dirs) != basename(rd)]
      if (!length(rate_dirs)) {
        loge("No rate* subdirectories found under subjective run %s", rd)
        next
      }
      logi("Subjective run %s has rate dirs: %s",
           rd, paste(basename(rate_dirs), collapse = ", "))

      for (rdir in rate_dirs) {
        bn <- basename(rdir)
        if (!startsWith(bn, "rate")) next
        noise_str <- sub("^rate", "", bn)
        if (noise_str == "") noise_str <- NA_character_
        df <- read_metrics_single(rdir, setup = st, noise_level = noise_str)
        if (!is.null(df) && nrow(df)) {
          rows[[length(rows) + 1L]] <- df
        } else {
          loge("No metrics read from subjective rate dir %s", rdir)
        }
      }
    } else {
      df <- read_metrics_single(rd, setup = st, noise_level = NA_character_)
      if (!is.null(df) && nrow(df)) {
        rows[[length(rows) + 1L]] <- df
      } else {
        loge("No metrics read from run dir %s", rd)
      }
    }
  }

  if (!length(rows)) {
    loge("load_all_metrics: no metrics rows accumulated.")
    return(NULL)
  }
  dplyr::bind_rows(rows)
}

# ----- gain delta computation -----

compute_gain_deltas <- function(df_raw) {
  df2 <- df_raw %>%
    dplyr::filter(!is.na(gain), !is.na(step))

  if (!nrow(df2)) {
    loge("All gains or steps are NA after filtering.")
    return(NULL)
  }

  # Average across seeds/runs within (setup, noise_level, step)
  df_grouped <- df2 %>%
    dplyr::group_by(setup, noise_level, step) %>%
    dplyr::summarise(gain = mean(gain, na.rm = TRUE), .groups = "drop")

  logi("Grouped metrics have %d rows across setups: %s",
       nrow(df_grouped),
       paste(sort(unique(df_grouped$setup)), collapse = ", "))

  perf <- df_grouped %>%
    dplyr::filter(setup == "perfect") %>%
    dplyr::select(step, gain_perf = gain)

  if (!nrow(perf)) {
    loge("No 'perfect' regime rows in aggregated data.")
    return(NULL)
  }

  df_join <- df_grouped %>%
    dplyr::left_join(perf, by = "step")

  if (any(is.na(df_join$gain_perf))) {
    missing_steps <- sort(unique(df_join$step[is.na(df_join$gain_perf)]))
    loge("Perfect regime is missing gain for steps: %s",
         paste(missing_steps, collapse = ", "))
  }

  # Change in gain relative to perfect.
  # If perfect is best, this will typically be <= 0 (negative bars),
  # with perfect itself at 0.
  df_join <- df_join %>%
    dplyr::mutate(
      setup_norm = tolower(gsub(" ", "_", setup)),
      gain_delta = ifelse(setup == "perfect", 0, gain - gain_perf),
      setup_label = dplyr::case_when(
        setup_norm == "perfect"                        ~ "Ideal",
        setup_norm == "expert"                         ~ "Expert",
        setup_norm == "subjective"                     ~ "Subjective",
        setup_norm %in% c("machine_annotation",
                          "detected")                  ~ "Machine Annotation",
        setup_norm == "automated_detection__llm"       ~ "LLM Discovery",
        setup_norm == "automated_detection__clip"      ~ "CLIP Discovery",
        TRUE ~ stringr::str_to_title(setup)
      )
    )

  df_join
}

# ----- plotting -----

plot_gain_deltas <- function(df, out_file) {
  if (!nrow(df)) {
    loge("No data to plot.")
    return(invisible(NULL))
  }

  # Drop 'perfect' from the plotted series; we only want it as the zero baseline
  if ("setup" %in% names(df)) {
    df <- df %>% dplyr::filter(setup != "perfect")
  } else {
    df <- df %>% dplyr::filter(setup_label != "Ideal")
  }
  if (!nrow(df)) {
    loge("No non-perfect setups to plot.")
    return(invisible(NULL))
  }

  df <- df %>%
    dplyr::mutate(
      gain_delta_pp = 100 * gain_delta
    )

  # ---- series ordering: by ABSOLUTE delta at largest step ----
  all_steps <- df$step[is.finite(df$step)]
  if (!length(all_steps)) {
    loge("No finite step values for ordering.")
    return(invisible(NULL))
  }

  max_step <- max(all_steps, na.rm = TRUE)
  logi("Ordering series by |gain_delta| at largest step k = %s", as.character(max_step))

  ref <- df %>%
    dplyr::filter(step == max_step) %>%
    dplyr::group_by(setup_label) %>%
    dplyr::summarise(
      gain_delta_pp = mean(gain_delta_pp, na.rm = TRUE),
      .groups = "drop"
    )

  # Sort from smallest |delta| to largest |delta|
  ref <- ref[order(abs(ref$gain_delta_pp)), , drop = FALSE]

  base_order <- ref$setup_label
  all_labels <- sort(unique(df$setup_label))
  missing    <- setdiff(all_labels, base_order)
  new_levels <- c(base_order, missing)

  df$setup_label <- factor(df$setup_label, levels = new_levels)

  logi("Series order (from smallest to largest |delta| at k=%s): %s",
       as.character(max_step), paste(new_levels, collapse = ", "))

  # ---- restrict x-axis to 1, 2, and max interventions and relabel ----
  keep_steps <- sort(unique(c(1L, 2L, max_step)))
  df <- df %>%
    dplyr::filter(step %in% keep_steps) %>%
    dplyr::mutate(
      step_factor = factor(
        step,
        levels = keep_steps,
        labels = c(
          "1",
          "2",
          paste0("5")
        )[seq_along(keep_steps)]
      )
    )

  if (!nrow(df)) {
    loge("No data to plot after filtering steps.")
    return(invisible(NULL))
  }

  # ---- y-axis: negative deltas up to 0, dashed 0 line ----
  y_min <- suppressWarnings(min(df$gain_delta_pp, na.rm = TRUE))
  if (!is.finite(y_min)) y_min <- -1
  y_max <- 0

  logi("Y-axis range for plot: [%.3f, %.3f] (percentage points)", y_min, y_max)

    p <- ggplot(df, aes(x = step_factor, y = gain_delta_pp, fill = setup_label)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(
      x    = "Intervention Step",
      y    = "Change in accuracy (%)"
      # no fill -> no legend title text
    ) +
    .base_theme +
    guides(fill = guide_legend(nrow = 3, byrow = TRUE, label.hjust = 0)) +
    scale_y_continuous(limits = c(y_min, y_max), expand = c(0, 0.02sAdditio))

  # Big figure so nothing gets clipped
  ggsave(out_file, plot = p, width = 16, height = 12, dpi = 300)
  logi("Wrote plot: %s", out_file)
}

# ----- main -----

main <- function() {
  args <- parse_args()

  logi("Working directory (getwd) = %s", getwd())

  repo_root    <- normalizePath(args$repo_root, mustWork = TRUE)
  results_root <- normalizePath(args$results_root, mustWork = FALSE)

  logi("repo_root    = %s", repo_root)
  logi("results_root = %s", results_root)
  logi("run_name     = %s", args$run_name)
  logi("outdir       = %s", args$outdir)

  runs <- find_run_dirs(results_root, args$run_name)
  if (!length(runs)) {
    loge("No run directories under %s matching %s*",
         results_root, args$run_name)
    quit(status = 1)
  }

  run_setups <- data.frame(
    run_dir = runs,
    setup   = vapply(runs, extract_setup, FUN.VALUE = character(1)),
    stringsAsFactors = FALSE
  )
  run_setups <- run_setups[
    !is.na(run_setups$setup) & nzchar(run_setups$setup),
    ,
    drop = FALSE
  ]

  if (!nrow(run_setups)) {
    loge("None of the run directories have '__regime-<setup>' in their names.")
    quit(status = 1)
  }

  logi("Using %d run directories with parsed setups.", nrow(run_setups))
  logi("Setups present: %s",
       paste(sort(unique(run_setups$setup)), collapse = ", "))

  if (!any(run_setups$setup == "perfect")) {
    loge("No 'perfect' regime run found (expected directory name containing '__regime-perfect').")
    quit(status = 1)
  }

  out_root <- normalizePath(args$outdir, mustWork = FALSE)
  if (!dir.exists(out_root)) {
    logi("Creating outdir: %s", out_root)
    dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
  }
  write.csv(run_setups,
            file = file.path(out_root, "RUNS_FOUND.csv"),
            row.names = FALSE)

  df_raw <- load_all_metrics(run_setups)
  if (is.null(df_raw) || !nrow(df_raw)) {
    loge("No metrics tables could be constructed from the selected runs.")
    quit(status = 1)
  }
  logi("Loaded raw metrics: %d rows", nrow(df_raw))

  # Always write full raw metrics for debugging
  write.csv(df_raw,
            file = file.path(out_root, "raw_metrics_all.csv"),
            row.names = FALSE)

  # Hard-coded subjective filter: KEEP only specified rates
  if (length(SUBJECTIVE_RATES_KEEP) > 0) {
    logi("Restricting subjective noise levels to: %s",
         paste(SUBJECTIVE_RATES_KEEP, collapse = ", "))

    before_n <- nrow(df_raw)
    df_raw <- df_raw %>%
      dplyr::filter(
        setup != "subjective" |
          (!is.na(noise_level) & noise_level %in% SUBJECTIVE_RATES_KEEP)
      )
    after_n <- nrow(df_raw)
    logi("Rows before subjective filter: %d, after: %d",
         before_n, after_n)
  }

  df <- compute_gain_deltas(df_raw)
  if (is.null(df) || !nrow(df)) {
    loge("No gain deltas available after merging with perfect regime.")
    quit(status = 1)
  }

  write.csv(df,
            file = file.path(out_root, "gain_vs_perfect_all.csv"),
            row.names = FALSE)

  out_file <- file.path(out_root,
                        paste0("gain_delta_vs_perfect_", args$run_name, ".pdf"))
  plot_gain_deltas(df, out_file)

  logi("[done] Outputs written under: %s", out_root)
}

if (identical(environment(), globalenv())) {
  tryCatch(
    main(),
    error = function(e) {
      loge("%s", e$message)
      quit(status = 1)
    }
  )
}
