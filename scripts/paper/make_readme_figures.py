"""Draw the example result plots of the README (docs/assets/*.png) from the paper's installed runs.

Uses the general plotting functions of `concept_benchmark.evaluation`, so the same calls work on the results of
any model run through the pipelines.

    python scripts/paper/make_readme_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import BALANCED_RULE, INTERVENTION_RECORDS, PAPER_RESULTS, REPO, SUDOKU_CELLS, filename_fields

ASSETS_DIR = REPO / "docs/assets"
CONCEPT_SETS = {"true": "true_concepts", "human": "human_concepts", "machine": "machine_annotation",
                "llm": "llm_concepts", "clip": "clip_concepts"}


def _read_robot_runs() -> pd.DataFrame:
    """Every installed balanced-rule run as one table (one row per run and budget)."""
    frames = []
    for path in sorted(BALANCED_RULE.glob("*__isrc-*__results.csv")):
        fields = filename_fields(path)
        is_label_free_cbm = fields["concepts"] in ("machine", "llm", "clip") and fields["arch"] == "cbm"
        if is_label_free_cbm and fields.get("enc") != "koh595":
            continue  # label-free CBMs are scored with the percentile encoding
        frame = pd.read_csv(path)[["budget", "accuracy"]]
        frames.append(frame.assign(
            model_family=fields["arch"], concept_source=CONCEPT_SETS[fields["concepts"]],
            intervention_source=fields["isrc"], seed=int(fields["seed"]),
        ))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    from concept_benchmark.evaluation import (
        plot_alignment_comparison,
        plot_answer_reliance,
        plot_automation,
        plot_concept_report,
        plot_confidence,
        plot_intervention_curve,
        plot_intervention_heatmap,
    )

    def save(fig, name: str) -> None:
        fig.savefig(ASSETS_DIR / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {ASSETS_DIR / name}")

    runs = _read_robot_runs()
    dnn = [
        float(frame.loc[frame["model"] == "dnn", "accuracy"].iloc[0])
        for frame in map(pd.read_csv, sorted(BALANCED_RULE.glob("*concepts-human__arch-cbm-and-dnn__*")))
    ]
    is_human_perfect = (runs["concept_source"] == "human_concepts") & (runs["intervention_source"] == "perfect")
    fig, _ = plot_intervention_curve(runs[is_human_perfect], group="model_family", baseline_accuracy=dnn)
    save(fig, "intervention_curve.png")

    fig, _ = plot_intervention_heatmap(runs[runs["intervention_source"] != "self"])
    save(fig, "intervention_heatmap.png")

    is_cbm = runs["model_family"] == "cbm"
    fig, _ = plot_answer_reliance(runs[is_cbm & runs["intervention_source"].isin(["self", "perfect"])], group="concept_source")
    save(fig, "answer_reliance.png")

    alignment = pd.read_csv(next((PAPER_RESULTS / "robot/alignment").glob("*__alignment.csv")))
    rows = [
        {"concepts": CONCEPT_SETS[r.concepts], "seed": r.seed, "model": model,
         "accuracy_before": getattr(r, f"k0_{suffix}") / 100, "accuracy_after": getattr(r, f"kmax_{suffix}") / 100}
        for r in alignment.itertuples()
        for model, suffix in (("CBM", "before"), ("Constrained CBM", "after"))
    ]
    fig, _ = plot_alignment_comparison(pd.DataFrame(rows))
    save(fig, "alignment.png")

    records = np.load(next(INTERVENTION_RECORDS.glob(
        "*concepts-human__arch-cbm__isrc-llm*__budget-1__seed-1014__records.npz")))
    fig, _ = plot_concept_report(
        records["mask"], records["C_pred"], records["C_answer"], records["C_true"], list(records["concept_names"])
    )
    save(fig, "concept_report.png")

    cells = sorted(SUDOKU_CELLS.glob("sudoku__arch-cbm__res-18px__tau-0.95__*__interventions.csv"))
    sudoku = pd.concat([pd.read_csv(c).assign(seed=i) for i, c in enumerate(cells)], ignore_index=True)
    fig, _ = plot_automation(sudoku, n_instances=200, n_concepts=27)
    save(fig, "automation.png")

    seed = 171
    confidence = np.load(PAPER_RESULTS / f"sudoku/confidence/sudoku__arch-cbm__res-18px__seed-{seed}__confidence.npz")
    cell = pd.read_csv(next(c for c in cells if f"seed-{seed}" in c.name))
    threshold = float(cell.sort_values("budget")["abstention_threshold"].iloc[0])
    fig, _ = plot_confidence(confidence["p_test"], confidence["y_test"], threshold)
    save(fig, "confidence.png")


if __name__ == "__main__":
    main()
