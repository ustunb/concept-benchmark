"""Write the two tables on confidence in sudoku automation and print the error breakdown cited in the text.

Reads the confidence files written by `scripts/sudoku_pipeline.py --stages diagnose` (results/paper/sudoku/confidence)
and the installed intervention cells (results/paper/sudoku/cells), all seeds, selective-accuracy target tau:

  --classwise-out   boards on which a model predicts under the pipeline's single abstention threshold and under one
                    threshold per predicted class, both fitted on validation and scored on test probabilities;
  --confirmed-out   each model's P(valid) for a valid board once all concepts are set to their true values, and the share
                    of deferred boards that checking every concept recovers, on the seeds where the model requires less
                    or more confidence than that (budget = all concepts, 18 px).

Printed: label errors by kind, the concept detector's misses and false alarms, and the share of boards accepted as valid
by number of violated constraints.

    PYTHONPATH=. python scripts/paper/make_sudoku_confidence_tables.py \\
        --classwise-out ../concept-benchmark-paper/tables/sudoku_classwise_threshold.tex \\
        --confirmed-out ../concept-benchmark-paper/tables/sudoku_confirmed_confidence.tex
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

from _common import PAPER, add_results_root, read_budget_rows, use_results_root
from concept_benchmark.evaluation import (
    abstention_threshold,
    classwise_thresholds,
    decision_threshold,
    selective_at,
    selective_at_classwise,
)

ARCHS = [
    ("cbm", "\\CBM{}"),
    ("cem", "\\CEM{}"),
    ("probcbm", "\\ProbCBM{}"),
    ("ecbm", "\\ECBM{}"),
]
RESOLUTIONS = (50, 18)


def seeds(res: int) -> list[int]:
    return sorted(
        int(f.name.split("seed-")[1].split("__")[0])
        for f in PAPER.sudoku_confidence.glob(
            f"sudoku__arch-cbm__res-{res}px__seed-*__confidence.npz"
        )
    )


def confidence(arch: str, res: int, seed: int):
    return np.load(
        PAPER.sudoku_confidence
        / f"sudoku__arch-{arch}__res-{res}px__seed-{seed}__confidence.npz"
    )


def cell(arch: str, res: int, seed: int, tau: str) -> list[dict]:
    path = (
        PAPER.sudoku_cells
        / f"sudoku__arch-{arch}__res-{res}px__tau-{tau}__threshold-per-budget__seed-{seed}__interventions.csv"
    )
    return read_budget_rows(path)


def classwise(tau: float) -> dict[tuple[int, str], tuple[float, float]]:
    """(resolution, architecture) -> mean share of test boards kept (percent) under the single / per-class rule."""
    out = {}
    for res in RESOLUTIONS:
        for arch, _ in ARCHS:
            single, per_class = [], []
            for seed in seeds(res):
                d = confidence(arch, res, seed)
                decision, _ = decision_threshold(d["y_val"], d["p_val"])
                t, _ = abstention_threshold(d["y_val"], d["p_val"], tau, decision)
                single.append(
                    0.0
                    if t is None
                    else 100 * selective_at(d["y_test"], d["p_test"], t, decision)[1]
                )
                t_pos, t_neg = classwise_thresholds(
                    d["y_val"], d["p_val"], tau, decision
                )
                per_class.append(
                    100
                    * selective_at_classwise(
                        d["y_test"], d["p_test"], t_pos, t_neg, decision
                    )[1]
                )
            out[(res, arch)] = (st.mean(single), st.mean(per_class))
    return out


def confirmed(tau: str, res: int = 18) -> dict[str, dict]:
    """Per architecture: confidence in a valid board with all concepts true; recovery where the model needs less / more."""
    out = {}
    for arch, _ in ARCHS:
        values, less, more = [], [], []
        for seed in seeds(res):
            d = confidence(arch, res, seed)
            value = float(np.median(d["p_test_true_concepts"][d["y_test"] == 1]))
            values.append(value)
            rows = cell(arch, res, seed, tau)
            required = 1.0 - float(rows[0]["abstention_threshold"])
            before, after = (
                float(rows[0]["coverage_after"]),
                float(rows[-1]["coverage_after"]),
            )
            if before >= 1.0:
                continue  # nothing deferred
            (less if value > required else more).append(
                100 * (after - before) / (1.0 - before)
            )
        out[arch] = {"confidence": st.median(values), "less": less, "more": more}
    return out


def print_error_breakdown() -> None:
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (9, 27)]
    for res in RESOLUTIONS:
        print(f"\n{res} px, test boards of all seeds")
        for arch, _ in ARCHS:
            p, y, C, Cp = (
                np.concatenate([confidence(arch, res, s)[k] for s in seeds(res)])
                for k in ("p_test", "y_test", "C_test", "Cp_test")
            )
            called_valid, violated, flagged = p >= 0.5, C == 0, Cp < 0.5
            n_violated = violated.sum(axis=1)
            accepted = " ".join(
                f"{100 * called_valid[(n_violated >= a) & (n_violated <= b)].mean():3.0f}%"
                if ((n_violated >= a) & (n_violated <= b)).any()
                else "   -"
                for a, b in bins
            )
            print(
                f"  {arch:8s} invalid boards called valid {100 * called_valid[y == 0].mean():4.1f}% | valid boards called invalid {100 * (~called_valid[y == 1]).mean():4.1f}%"
                f" | detector misses {100 * (violated & ~flagged).sum() / violated.sum():4.1f}% of violations, flags {100 * (~violated & flagged).sum() / (~violated).sum():5.2f}% of satisfied constraints"
                f" | accepted with 1/2/3/4/5/6-8/9+ violations: {accepted}"
            )


def spread(values: list[float]) -> str:
    if not values:
        return "--"
    lo, hi = round(min(values)), round(max(values))
    return f"{lo}\\%" if lo == hi else f"{lo}--{hi}\\%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="0.95")
    ap.add_argument("--classwise-out", type=Path, default=None)
    ap.add_argument("--confirmed-out", type=Path, default=None)
    add_results_root(ap)
    args = ap.parse_args()
    use_results_root(args)

    kept = classwise(float(args.tau))
    print(
        f"boards kept at tau={args.tau}: single threshold / one threshold per predicted class"
    )
    for (res, arch), (a, b) in kept.items():
        print(f"  {res}px {arch:8s} {a:5.1f}% / {b:5.1f}%")
    if args.classwise_out:
        lines = [
            r"\begin{tabular}{@{}llrr@{}}",
            r"\toprule",
            r"\textheader{Resolution} & \textheader{Model} & \textheader{Single threshold} & \textheader{Threshold per class} \\",
        ]
        for res in RESOLUTIONS:
            lines.append(r"\midrule")
            for i, (arch, macro) in enumerate(ARCHS):
                first = (
                    rf"\multirow{{4}}{{*}}{{\textds{{{res}\,px}}}}" if i == 0 else ""
                )
                lines.append(
                    f"{first} & {macro} & {kept[(res, arch)][0]:.1f}\\% & {kept[(res, arch)][1]:.1f}\\% \\\\"
                )
        lines += [r"\bottomrule", r"\end{tabular}"]
        args.classwise_out.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.classwise_out}")

    summary = confirmed(args.tau)
    print(
        "\nconfidence in a valid board with every concept true (18 px), and deferred boards recovered at budget = all concepts"
    )
    for arch, _ in ARCHS:
        s = summary[arch]
        print(
            f"  {arch:8s} {s['confidence']:.3f} | requires less: {len(s['less'])} seeds, recovered {[round(v) for v in s['less']]} | requires more: {len(s['more'])} seeds, recovered {[round(v) for v in s['more']]}"
        )
    if args.confirmed_out:
        lines = [
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"& & \multicolumn{2}{c}{\textheader{Requires less}} & \multicolumn{2}{c}{\textheader{Requires more}} \\",
            r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
            r"\textheader{Model} & \textheader{Confidence} & seeds & recovered & seeds & recovered \\",
            r"\midrule",
        ]
        for arch, macro in ARCHS:
            s = summary[arch]
            value = (
                f"{s['confidence']:.3f}"
                if s["confidence"] > 0.995
                else f"{s['confidence']:.2f}"
            )
            lines.append(
                f"{macro} & {value} & {len(s['less'])} & {spread(s['less'])} & {len(s['more'])} & {spread(s['more'])} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        args.confirmed_out.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.confirmed_out}")

    print_error_breakdown()


if __name__ == "__main__":
    main()
