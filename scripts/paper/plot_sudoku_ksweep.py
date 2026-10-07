"""Net work automated of the CBM against the intervention budget, on validation and on test (the budget sweep).

Reads results/paper/sudoku/ksweep (one CSV per seed, resolution and split, written by `sweep_sudoku_budgets.py` over
the budgets 1, 2, 3, 5, 8, 13, max) and draws one panel per resolution: mean ± SE over seeds, the best budget of each
curve marked. Prints, per seed, the budget chosen on validation, the best budget on test, and the net work lost on
test by using the validation choice (the regret).

    python scripts/paper/plot_sudoku_ksweep.py --out ../concept-benchmark-paper/figures/sudoku_ksweep.pdf
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import PAPER, add_results_root, use_results_root  # noqa: E402

BUDGETS = (0, 1, 2, 3, 5, 8, 13, 27)
RESOLUTIONS = (50, 18)
TAU = "0.99"
COLORS = {"validation": "#4D4D4D", "test": "#3B6FB6"}


def curves(res: int, split: str) -> dict[int, list[float]]:
    """seed -> net work automated (percent) at each budget."""
    out = {}
    for f in sorted(
        PAPER.sudoku_ksweep.glob(
            f"sudoku__arch-cbm__res-{res}px__tau-{TAU}__{split}__seed-*__ksweep.csv"
        )
    ):
        seed = int(f.stem.split("seed-")[1].split("__")[0])
        rows = {
            int(r["budget"]): float(r["net_work"]) for r in csv.DictReader(f.open())
        }
        out[seed] = [100 * rows[k] for k in BUDGETS]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    add_results_root(ap)
    args = ap.parse_args()
    use_results_root(args)
    fig, axes = plt.subplots(
        1, len(RESOLUTIONS), figsize=(4.4 * len(RESOLUTIONS), 3.4), sharey=True
    )
    x = range(len(BUDGETS))
    for ax, res in zip(axes, RESOLUTIONS):
        data = {split: curves(res, split) for split in ("validation", "test")}
        seeds = sorted(set(data["validation"]) & set(data["test"]))
        if not seeds:
            raise SystemExit(f"{res}px: no budget sweep found in {PAPER.sudoku_ksweep}")
        for split, color in COLORS.items():
            values = [data[split][s] for s in seeds]
            mean = [st.mean(v[i] for v in values) for i in x]
            se = [
                st.stdev(v[i] for v in values) / len(values) ** 0.5
                if len(values) > 1
                else 0
                for i in x
            ]
            ax.plot(
                x,
                mean,
                "--" if split == "validation" else "-",
                color=color,
                marker="o",
                ms=3.5,
                label=split,
            )
            ax.fill_between(
                x,
                [m - e for m, e in zip(mean, se)],
                [m + e for m, e in zip(mean, se)],
                color=color,
                alpha=0.12,
            )
            best = max(x, key=lambda i: mean[i])
            ax.annotate(
                f"k*={BUDGETS[best] if BUDGETS[best] != 27 else 'max'}",
                (best, mean[best]),
                textcoords="offset points",
                xytext=(0, 9 if split == "test" else -14),
                ha="center",
                fontsize=8,
                color=color,
            )
        regrets, gains, same = [], [], 0
        for s in seeds:
            v, t = data["validation"][s], data["test"][s]
            k_val, k_test = max(x, key=lambda i: v[i]), max(x, key=lambda i: t[i])
            regrets.append(t[k_test] - t[k_val])
            gains.append(t[k_val] - t[0])
            same += k_val == k_test
        print(
            f"{res}px: n={len(seeds)}  validation-chosen budget: regret on test {st.mean(regrets):.2f} points "
            f"(max {max(regrets):.2f}), gain over k=0 {st.mean(gains):.2f} points, same budget as test on {same}/{len(seeds)} seeds"
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(b) if b != 27 else "max" for b in BUDGETS])
        ax.set_title(f"CBM, {res} px, τ = {TAU}")
        ax.set_xlabel("Intervention budget k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Net work automated (%)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
