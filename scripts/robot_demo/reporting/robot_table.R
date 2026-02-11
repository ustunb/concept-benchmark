packages = c("reticulate", "tidyverse", "stringr", "kableExtra")

for (pkg in packages) {
    library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
}
options(dplyr.width = Inf, dplyr.print_max = 1e9)
options(stringsAsFactors = FALSE)


EMPTY_TEX_STRING = "---"

robot_results <- "./results/robot_demo_results.csv"
table_tex_path <- "~/Dropbox/Apps/Overleaf/concept-benchmark/tables/robot_cbm.tex"

TAU = 0.2
metric_order = c('accuracy', 'predictions_intervened_on', 'concept_confirmations_per_instance', 'predictions_changed')
model_order = c("dnn", "cbm_no_int", "cbm_with_int_1", "cbm_with_int_3", "cbm_with_int_max")
missing_order = c("none", "mcar", "mnar")
DATASET_TITLES <- c(
    "ideal_none" = "\\robotIdeal{}",
    "ideal_mcar" = "\\robotIdealMCAR{}",
    "ideal_mnar" = "\\robotIdealMNAR{}",
    "subconcept_none" = "\\robotSubconcept{}",
    "subconcept_mcar" = "\\robotSubconceptMCAR{}",
    "subconcept_mnar" = "\\robotSubconceptMNAR{}"
)

df <- read_csv(robot_results, show_col_types = FALSE)
df <- df %>% mutate(model = ifelse((model == "cbm_with_int_7") | (model == "cbm_with_int_12"), "cbm_with_int_max", model))
df <- df %>% filter(model %in% model_order)
df$model <- factor(df$model, levels = model_order)
df$concept_missing_mech <- factor(df$concept_missing_mech, levels = missing_order)
df <- df %>%
    # mutate(model = ifelse(is.na(budget), model, paste0(model, "_", budget))) %>%
    filter((threshold == TAU) | is.na(threshold)) %>%
    select(data_name, concept_missing_mech, model, metric, value)

# Calculate concept confirmations per instance
df_wide <- df %>%
    pivot_wider(names_from = metric, values_from = value)

df_wide <- df_wide %>%
    mutate(concept_confirmations_per_instance =
               ifelse(!is.na(total_concept_confirmations) &
                        !is.na(predictions_intervened_on) &
                        predictions_intervened_on > 0,
                      total_concept_confirmations / predictions_intervened_on,
                      NA_real_))

df <- df_wide %>%
    pivot_longer(
        cols = c(accuracy, predictions_intervened_on,
                 total_concept_confirmations, predictions_changed,
                 concept_confirmations_per_instance),
        names_to = "metric",
        values_to = "value"
    ) %>%
    filter(!is.na(value)) %>%
    filter(metric %in% metric_order)

table_stats_df <- df %>%
    mutate(svalue_pct = sprintf("%1.1f\\%%", 100 * value),
        svalue_dec = sprintf("%1.2f", value),
        svalue_int = sprintf("%d", round(value))) %>%
    mutate(svalue = case_when(
        metric == "accuracy" ~ svalue_pct,
        metric == "concept_confirmations_per_instance" ~ svalue_dec,
        TRUE ~ svalue_int
    )) %>%
    select(-svalue_pct, -svalue_dec, -svalue_int)

cells_df <- table_stats_df %>%
    arrange(model, metric) %>%
    select(-value) %>%
    pivot_wider(
        names_from = c("metric"),
        # names_from = concept_missing_mech,
        values_from = svalue,
        values_fill = EMPTY_TEX_STRING
    )

table_df <- cells_df %>%
    group_by(data_name, model, concept_missing_mech) %>%
    unite(cell_str, sep = "\\\\", metric_order) %>%
    mutate(cell_str = sprintf("\\cell{r}{%s}\n", cell_str)) %>%
    ungroup() %>%
    arrange(data_name, model) %>%
    pivot_wider(
        names_from = c("model"),
        values_from = cell_str,
        names_sort = FALSE,
        # names_glue = "{model}",
    ) %>%
    # fill na values
    replace_na(list(dnn="\\cell{r}{---\\\\---\\\\---\\\\---}"))

kable_df <- table_df %>%
    mutate(data_name = paste0(data_name, "_", concept_missing_mech)) %>%
    mutate(data_name = recode(data_name, !!!DATASET_TITLES)) %>%
    relocate(data_name , .before = everything()) %>%
    select(-concept_missing_mech) %>%
    mutate(metrics = "\\robotAllMetrics{}") %>%
    relocate(metrics , .after = data_name)

top_headers <- c(" " = 3, "CBM" = 4)
mid_headers <- c("Dataset", "Metrics", "DNN", "No Int.", "w/ Int. ($k=1$)", "w/ Int. ($k=3$)", "w/ Int. (max)")

table <- kable_df %>%
    kable(
            booktabs = TRUE,
            escape = FALSE,
            col.names = mid_headers,
            format = "latex",
            table.envir = NULL,
            linesep = "\\midrule",
            align = c("l", "l", rep("r", 6))
        ) %>%
    add_header_above(header = top_headers, bold=FALSE, escape=FALSE) %>%
    row_spec(0, bold = TRUE)

cat(table, file = table_tex_path, sep = "\n")