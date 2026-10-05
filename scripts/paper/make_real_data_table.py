"""Write the real-dataset table (Derm7pt and CUB-25: accuracy by concept set and intervention budget).

Reads `results/paper/real_datasets/{derm7pt,cub}.csv`, written by
`experiments/real_data.py`, and writes two tabulars side by side: mean accuracy over runs at each
budget, and the change in accuracy at k = max.

    python scripts/paper/make_real_data_table.py --out ../concept-benchmark-paper/tables/real_datasets.tex
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from _common import PAPER_RESULTS

RESULTS = PAPER_RESULTS / "real_datasets"
DATASETS = [("derm7pt", r"\textds{Derm7pt}"), ("cub", r"\textds{CUB}")]
CONCEPTS = [
    (("clinician", "ground_truth"), "human-annotated"),
    (("label_free",), "label-free"),
]


def tabular(name: str, title: str) -> list[str]:
    rows = list(csv.DictReader((RESULTS / f"{name}.csv").open()))
    budgets = list(dict.fromkeys(r["budget"] for r in rows))
    head = " & ".join(
        r"$k{=}\text{max}$" if b == "max" else f"$k{{=}}{b}$" for b in budgets
    )
    lines = [
        rf"\begin{{tabular}}{{@{{}}l{'r' * (len(budgets) + 1)}@{{}}}}",
        r"\toprule",
        rf"{title} & {head} & \textheader{{Change}} \\",
        r"\midrule",
    ]
    for keys, label in CONCEPTS:
        runs = [r for r in rows if r["concepts"] in keys]
        means = [
            100 * st.mean(float(r["accuracy"]) for r in runs if r["budget"] == b)
            for b in budgets
        ]
        change = means[-1] - means[0]
        print(
            f"{name:8s} {label:16s} runs={len({r['seed'] for r in runs})} "
            + " ".join(f"{m:5.1f}" for m in means)
            + f"  change {change:+.1f}"
        )
        lines.append(
            f"{label} & "
            + " & ".join(f"{m:.1f}\\%" for m in means)
            + f" & ${'+' if change >= 0 else '-'}{abs(change):.1f}\\%$ \\\\"
        )
    return lines + [r"\bottomrule", r"\end{tabular}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    blocks = ["\n".join(tabular(name, title)) for name, title in DATASETS]
    args.out.write_text("\n\\quad\n".join(blocks) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
