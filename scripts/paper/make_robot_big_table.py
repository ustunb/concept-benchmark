"""Write the robots big table (accuracy by concept set x intervention source x architecture x budget).

Reads the installed results of one labeling rule and writes the table in the paper's layout: rows = concept set x
architecture, column blocks = perfect / expert / llm interventions at k = 0, 1, 3, max. Cells are mean ± SE over
seeds (accuracy, %); every cell must have all seeds of its rule (balanced: 1014-1023, results/paper/robot/balanced_rule;
sparse: 1014-1017, results/paper/robot/grid). LLM interventions use the Gemini 2.5 Flash-Lite caches at 224 px. Under
the balanced rule, interventions on the label-free CBM set a concept to its 5th/95th-percentile score.

    python scripts/paper/make_robot_big_table.py --rule balanced --out ../concept-benchmark-paper/tables/robot_big_table.tex
    python scripts/paper/make_robot_big_table.py --rule sparse --out ../concept-benchmark-paper/tables/robot_big_table_sparse.tex
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RULES = {  # rule -> (results folder, seeds per cell, table label, caption suffix)
    "balanced": (REPO / "results/paper/robot/balanced_rule", 10, "Table::BigTable", ""),
    "sparse": (REPO / "results/paper/robot/grid", 4, "Table::BigTableSparse", " under the sparse rule"),
}
AUTOMATED = {"machine_annotation", "llm_concepts", "clip_concepts"}
CONCEPT_SETS = [
    ("ground_truth", "true\\_concepts", 7),
    ("human_concepts", "human\\_concepts", 12),
    ("machine_annotation", "machine\\_annotation", 12),
    ("llm_concepts", "llm\\_concepts", 12),
    ("clip_concepts", "clip\\_concepts", 12),
]
ARCHS = [("cbm", "\\CBM{}"), ("cem", "\\CEM{}"), ("probcbm", "\\ProbCBM{}"), ("ecbm", "\\ECBM{}")]
SOURCES = ["perfect", "expert", "llm"]


def read_grid(rule: str) -> dict[tuple, list[list[float]]]:
    """(concept source, intervention source, family) -> per-seed accuracies at k = 0, 1, 3, max."""
    cells = defaultdict(list)
    for f in sorted(RULES[rule][0].glob("*.csv")):
        fields = dict(kv.split("-", 1) for kv in f.stem.split("__")[1:-1])
        if "isrc" not in fields:
            continue  # section 1's CBM-and-DNN files
        if fields.get("isrc") == "llm" and fields.get("img") != "224px":
            continue  # 32 px caches and seed 1014's April caches are kept apart
        rows = sorted(csv.DictReader(f.open()), key=lambda r: int(r["budget"]))
        key = (rows[0]["concept_source"], rows[0]["intervention_source"], rows[0]["model_family"])
        if rule == "balanced" and key[2] == "cbm" and key[0] in AUTOMATED and fields.get("enc") != "koh595":
            continue  # label-free CBM: 5th/95th-percentile interventions, not hard 0/1
        cells[key].append([100 * float(r["accuracy"]) for r in rows])
    return cells


def cell(values: list[float], is_best: bool = False) -> str:
    mean = st.mean(values)
    se = st.stdev(values) / len(values) ** 0.5
    if is_best:
        return f"\\textbf{{{mean:.1f}\\%}}$\\boldsymbol{{\\pm}}$\\textbf{{{se:.1f}}}"
    return f"{mean:.1f}\\%$\\pm${se:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=sorted(RULES), default="balanced")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _, n_seeds, label, suffix = RULES[args.rule]
    cells = read_grid(args.rule)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Accuracy of concept-based models in the decision-support benchmark under each concept set and "
        r"intervention regime with $k$ interventions"
        rf"{suffix}{' (mean $' + chr(92) + 'pm$ SE, $n{=}4$ seeds)' if args.rule == 'sparse' else ''}.}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\resizebox{1.0\textwidth}{!}{",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}ll rrrr|rrrr|rrrr@{}}",
        r"\toprule",
        r"& & \multicolumn{12}{c}{\textbf{Intervention}} \\",
        r"\cmidrule(lr){3-14}",
        r"& & \multicolumn{4}{c|}{\textds{perfect}} & \multicolumn{4}{c|}{\textds{expert}} & \multicolumn{4}{c}{\textds{llm}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}",
        r"\textbf{Dataset} & \textbf{Model} & $k{=}0$ & $k{=}1$ & $k{=}3$ & max & $k{=}0$ & $k{=}1$ & $k{=}3$ & max & $k{=}0$ & $k{=}1$ & $k{=}3$ & max \\",
    ]
    for i, (source, name, m) in enumerate(CONCEPT_SETS):
        lines.append(r"\midrule")
        lines.append(rf"\multirow{{4}}{{*}}{{\cell{{l}}{{\textds{{{name}}}\\$m{{=}}{m}$}}}}")
        runs = {}
        for family, _ in ARCHS:
            for isrc in SOURCES:
                runs[family, isrc] = cells.get((source, isrc, family)) or []
                if len(runs[family, isrc]) != n_seeds:
                    raise SystemExit(f"cell {source} x {isrc} x {family}: {len(runs[family, isrc])} seeds, expected {n_seeds}")
        # bold the highest mean (as printed) in each concept set x intervention source block, once per row
        best = {
            isrc: max(round(st.mean([r[k] for r in runs[family, isrc]]), 1) for family, _ in ARCHS for k in range(4))
            for isrc in SOURCES
        }
        for family, macro in ARCHS:
            row = []
            for isrc in SOURCES:
                means = [round(st.mean([r[k] for r in runs[family, isrc]]), 1) for k in range(4)]
                first_best = means.index(best[isrc]) if best[isrc] in means else None  # earliest budget only
                row += [cell([r[k] for r in runs[family, isrc]], is_best=k == first_best) for k in range(4)]
            lines.append(f" & {macro:<11}& " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(CONCEPT_SETS) * len(ARCHS)} rows)")


if __name__ == "__main__":
    main()
