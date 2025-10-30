packages = c("reticulate", "tidyverse", "stringr", "kableExtra")

for (pkg in packages) {
    library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
}
options(dplyr.width = Inf, dplyr.print_max = 1e9)
options(stringsAsFactors = FALSE)


EMPTY_TEX_STRING = "---"

robot_results <- "./results/robot_demo_results.csv"
table_tex_path <- "~/Dropbox/Apps/Overleaf/concept-benchmark/tables/robot_cbm.tex"

TAU = 0.6
metric_order = c("accuracy", "predictions_intervened_on", "total_concept_edits_made")
model_order = c("dnn", "cbm_no_int", "cbm_with_int_1", "cbm_with_int_3")
DATASET_TITLES <- c(
    "ideal" = "\\robotIdeal{}",
    "subconcept" = "\\robotSubconcept{}"
)

df <- read_csv(robot_results, show_col_types = FALSE)
df$metric <- factor(df$metric, levels = metric_order)
df$model <- factor(df$model, levels = model_order)
df <- df %>%
    # mutate(model = ifelse(is.na(budget), model, paste0(model, "_", budget))) %>%
    filter((threshold == TAU) | is.na(threshold)) %>%
    select(data_name, model, metric, value)

table_stats_df <- df %>%
    mutate(svalue_pct = sprintf("%1.1f\\%%", 100 * value),
        svalue_dec = sprintf("%1.3f", value),
        svalue_int = sprintf("%d", round(value))) %>%
    mutate(svalue = ifelse(metric == "accuracy", svalue_dec,
                        ifelse(metric == "predictions_intervened_on", svalue_int,
                               svalue_int))) %>%
    select(-svalue_pct, -svalue_dec, -svalue_int)

cells_df <- table_stats_df %>%
    arrange(model, metric) %>%
    select(-value) %>%
    pivot_wider(
        names_from = metric,
        values_from = svalue
    )

cells_df[is.na(cells_df)] <- EMPTY_TEX_STRING

table_df <- cells_df %>%
    group_by(data_name, model) %>%
    unite(cell_str, sep = "\\\\", metric_order) %>%
    mutate(cell_str = sprintf("\\cell{r}{%s}\n", cell_str)) %>%
    ungroup() %>%
    arrange(data_name, model)

kable_df <- table_df %>%
    mutate(
        metrics = "\\robotMetrics{}",
        data_name = recode(data_name, !!!DATASET_TITLES)
    ) %>%
    pivot_wider(
        names_from = c(model),
        values_from = cell_str,
        names_sort = FALSE,
        names_glue = "{model}",
    )

kable_df[is.na(kable_df)] <- "\\cell{r}{---\\\\---\\\\---}"

top_headers <- c("Dataset", "Metrics", "DNN", "CBM No Int.", "CBM With Int. ($k=1$)", "CBM With Int. ($k=3$)")
# mid_headers <- c("Concept Noise", "Concept Missing", "Difficulty", "Intervened", "Before", "After")

table <- kable_df %>%
    kbl(escape = FALSE, toprule = '', align = "c", booktabs = TRUE, linesep = "\\midrule", 
        col.names = top_headers, format="latex")

cat(table, file = table_tex_path, sep = "\n")