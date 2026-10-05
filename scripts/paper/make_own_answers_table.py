"""Write the own-answers table: what interventions do when they supply the model's own answers or the true values.

For every concept set and architecture (balanced rule, seeds 1014-1023) it compares two installed intervention cells:
`isrc-self` (every selected concept is set to the value the model already predicts) and `isrc-perfect` (set to the true
value). A robot that is not intervened on keeps its prediction, so the change in accuracy is reported on the robots
intervened on: (accuracy at budget k - accuracy at k = 0) / share of robots intervened on, averaged over k = 1, 3, max
and seeds. The label-free CBM uses the 5th/95th-percentile encoding in both cells.

    python scripts/paper/make_own_answers_table.py --out ../concept-benchmark-paper/tables/own_answers.tex
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from _common import BALANCED_RULE, BALANCED_TAG, read_budget_rows

SEEDS = tuple(range(1014, 1024))
ARCHS = [
    ("cbm", "arch-cbm", "\\CBM{}"),
    ("cem", "arch-cem", "\\CEM{}"),
    ("probcbm", "arch-probcbm__mode-joint", "\\ProbCBM{}"),
    ("ecbm", "arch-ecbm", "\\ECBM{}"),
]
CONCEPT_SETS = [
    ("true", "true\\_concepts"),
    ("human", "human\\_concepts"),
    ("machine", "machine\\_annotation"),
    ("llm", "llm\\_concepts"),
    ("clip", "clip\\_concepts"),
]
AUTOMATED = {"machine", "llm", "clip"}
# the installed perfect cell of this run comes from a retrained model; its own-answer cell was run on the replaced
# model, so the pair is left out
EXCLUDED = {("probcbm", "true", 1014)}


def read_cell(
    arch_tag: str, concepts: str, isrc: str, seed: int, is_label_free: bool
) -> list[dict]:
    files = [
        f
        for f in BALANCED_RULE.glob(
            f"{BALANCED_TAG}__concepts-{concepts}__{arch_tag}__isrc-{isrc}__strategy-upto*__seed-{seed}__results.csv"
        )
        if "enc-" not in f.name or ("enc-koh595" in f.name) == is_label_free
    ]
    if len(files) != 1:
        raise SystemExit(
            f"{concepts} {arch_tag} {isrc} seed {seed}: expected one cell, found {[f.name for f in files]}"
        )
    rows = read_budget_rows(files[0])
    if len(rows) != 4:
        raise SystemExit(f"{files[0].name}: expected budgets 0, 1, 3, max")
    return rows


def change_on_intervened(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Per budget > 0: change in accuracy on the robots intervened on (points) and their share (percent)."""
    base = float(rows[0]["accuracy"])
    changes, shares = [], []
    for r in rows[1:]:
        share = float(r["predictions_intervened_on"]) / float(r["n"])
        if share > 0:
            changes.append(100 * (float(r["accuracy"]) - base) / share)
            shares.append(100 * share)
    return changes, shares


def summarize() -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    for concepts, _ in CONCEPT_SETS:
        for arch, arch_tag, _ in ARCHS:
            own, true, shares = [], [], []
            for seed in SEEDS:
                if (arch, concepts, seed) in EXCLUDED:
                    continue
                is_label_free = arch == "cbm" and concepts in AUTOMATED
                self_rows = read_cell(arch_tag, concepts, "self", seed, is_label_free)
                perfect_rows = read_cell(
                    arch_tag, concepts, "perfect", seed, is_label_free
                )
                if [r["predictions_intervened_on"] for r in self_rows] != [
                    r["predictions_intervened_on"] for r in perfect_rows
                ]:
                    raise SystemExit(
                        f"{concepts} {arch} seed {seed}: the two cells intervene on different numbers of robots"
                    )
                c, s = change_on_intervened(self_rows)
                own += c
                shares += s
                true += change_on_intervened(perfect_rows)[0]
            out[(concepts, arch)] = {
                "share": st.mean(shares),
                "own": st.mean(own),
                "true": st.mean(true),
            }
    return out


def signed(value: float) -> str:
    text = f"{value:+.1f}"
    if text in ("+0.0", "-0.0"):
        return "$0.0\\%$"
    return f"${text[0] if text[0] == '+' else '{-}'}{text[1:]}\\%$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summary = summarize()
    lines = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textheader{Change in accuracy upon interventions}} \\",
        r"\cmidrule(lr){3-4}",
        r"\textheader{Dataset} & \textheader{Model} & \textheader{Own concept predictions} & \textheader{True concept values} \\",
    ]
    for concepts, name in CONCEPT_SETS:
        lines.append(r"\midrule")
        for i, (arch, _, macro) in enumerate(ARCHS):
            row = summary[(concepts, arch)]
            first = rf"\multirow{{4}}{{*}}{{\textds{{{name}}}}}" if i == 0 else ""
            lines.append(
                f"{first} & {macro} & {signed(row['own'])} & {signed(row['true'])} \\\\"
            )
            print(
                f"{concepts:8s}{arch:8s} robots intervened on {row['share']:5.1f}%  own answers {row['own']:+6.1f}  true values {row['true']:+6.1f}"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
