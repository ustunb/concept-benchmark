"""Write the robots big table (accuracy by concept set x intervention source x architecture x budget).

Reads the installed grid (results/paper/robot/grid) and writes `tables/robot_big_table.tex` in the paper's layout:
rows = concept set x architecture, column blocks = perfect / expert / llm interventions at k = 0, 1, 3, max.
Cells are mean ± SE over seeds (accuracy, %). LLM interventions use the Gemini 2.5 Flash-Lite caches at 224 px
(seeds 1015-1017); all other cells use seeds 1014-1017.

    python scripts/make_robot_big_table.py --out ../concept-benchmark-paper/tables/robot_big_table.tex
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GRID = REPO / "results/paper/robot/grid"
CONCEPT_SETS = [
    ("ground_truth", "true\\_concepts", 7),
    ("human_concepts", "human\\_concepts", 12),
    ("machine_annotation", "machine\\_annotation", 12),
    ("llm_concepts", "llm\\_concepts", 12),
    ("clip_concepts", "clip\\_concepts", 12),
]
ARCHS = [("cbm", "\\CBM{}"), ("cem", "\\CEM{}"), ("probcbm", "\\ProbCBM{}"), ("ecbm", "\\ECBM{}")]
SOURCES = ["perfect", "expert", "llm"]


def read_grid() -> dict[tuple, list[list[float]]]:
    """(concept source, intervention source, family) -> per-seed accuracies at k = 0, 1, 3, max."""
    cells = defaultdict(list)
    for f in sorted(GRID.glob("*.csv")):
        fields = dict(kv.split("-", 1) for kv in f.stem.split("__")[1:-1])
        if fields.get("isrc") == "llm" and fields.get("img") != "224px":
            continue  # 32 px caches and seed 1014's April caches are kept apart
        rows = sorted(csv.DictReader(f.open()), key=lambda r: int(r["budget"]))
        key = (rows[0]["concept_source"], rows[0]["intervention_source"], rows[0]["model_family"])
        cells[key].append([100 * float(r["accuracy"]) for r in rows])
    return cells


def cell(values: list[float]) -> str:
    mean = st.mean(values)
    se = st.stdev(values) / len(values) ** 0.5
    return f"{mean:.1f}$\\pm${se:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cells = read_grid()

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Our benchmarks evaluate how accuracy changes across concept sets and intervention regimes, "
        r"separating the effects of concept specification, concept source, intervention source, and architecture "
        r"choice. We report accuracy (\%, mean $\pm$ SE over seeds $1014$--$1017$; \textds{llm} interventions over "
        r"seeds $1015$--$1017$) under the sparse rule.}",
        r"\label{Table::BigTable}",
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
        for family, macro in ARCHS:
            row = []
            for isrc in SOURCES:
                runs = cells.get((source, isrc, family))
                if not runs:
                    raise SystemExit(f"missing cell {source} x {isrc} x {family}")
                row += [cell([r[k] for r in runs]) for k in range(4)]
            lines.append(f" & {macro:<11}& " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(CONCEPT_SETS) * len(ARCHS)} rows)")


if __name__ == "__main__":
    main()
