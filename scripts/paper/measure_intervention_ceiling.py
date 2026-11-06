"""Measure how far interventions can lift the deferred sudoku boards: the label probability with every concept true.

For one architecture, resolution and seed, loads the stored model and the recognizer-inferred dataset, scales the label
probability and fits the abstention threshold as `automation_table` does, and scores every test board twice: with the
predicted concepts and with every concept set to its true value. A board that stays in the abstention band with every
concept true cannot be labeled at the target by any intervention. Writes one row per test board:

    python scripts/paper/measure_intervention_ceiling.py --arch cbm --cell-px 50 --seed 171 --tau 0.99 \
        --models results/paper/sudoku/models --data results/paper/sudoku/data/sudoku --out results/paper/sudoku/ceiling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401  (puts the repository root on the import path)
from concept_benchmark.evaluation.metrics import (
    PlattScaling,
    abstention_mask,
    abstention_threshold,
)
from concept_benchmark.ext.fileutils import load
from experiments.intervention import (
    ConceptInterventionRunner,
    ConceptualSafeguardsStrategy,
    InterventionConfig,
    _label_proba_on_rows,
    _prime_aligned_replay_context,
)

MODEL_KEY = {"cbm": "cs", "cem": "cem", "probcbm": "probcbm", "ecbm": "ecbm"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=sorted(MODEL_KEY), required=True)
    ap.add_argument("--cell-px", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = load(
        args.data
        / f"sudoku_image_n3_ns1000_mc9_px{args.cell_px}_seed{args.seed}"
        / "ocr_inferred_full_dataset.pkl"
    )
    data.sample(test_size=0.2, val_size=0.2, stratify=data.y, seed=args.seed)
    model = load(
        args.models
        / f"sudoku_{MODEL_KEY[args.arch]}_tabular_n3_mc9_px{args.cell_px}_seed{args.seed}.model"
    )
    model._random_state = args.seed

    runner = ConceptInterventionRunner(model)
    policy = ConceptualSafeguardsStrategy()
    probe = InterventionConfig(
        abstention_threshold=0.0, per_instance_budget=0, random_state=args.seed
    )
    val_y = np.asarray(data.validation.y).astype(int)
    val_prob = runner.run(policy, probe, data.validation).y_prob_after[:, 1]
    scale = PlattScaling().fit(val_prob, val_y)
    threshold, _ = abstention_threshold(val_y, scale(val_prob), args.tau)

    test = data.test
    prob_predicted = scale(runner.run(policy, probe, test).y_prob_after[:, 1])
    batch = runner._build_batch(
        dataset=test,
        concept_proba=None,
        concept_true=test.base_concepts,
        labels=None,
        instance_ids=None,
    )
    _prime_aligned_replay_context(model, test, batch)
    concepts_true = np.nan_to_num(batch.C_true.astype(float), nan=1.0)
    rows = np.arange(concepts_true.shape[0])
    prob_true = scale(
        _label_proba_on_rows(
            model, batch, concepts_true, np.ones_like(concepts_true, bool), rows
        )[:, 1]
    )

    is_scored = threshold is not None
    table = pd.DataFrame(
        {
            "arch": args.arch,
            "px": args.cell_px,
            "seed": args.seed,
            "tau": args.tau,
            "threshold": threshold if is_scored else np.nan,
            "y": np.asarray(test.y).astype(int),
            "n_violations": (concepts_true < 0.5).sum(axis=1),
            "prob_predicted": prob_predicted,
            "prob_true": prob_true,
            "is_deferred": abstention_mask(prob_predicted, threshold)
            if is_scored
            else True,
            "is_deferred_true": abstention_mask(prob_true, threshold)
            if is_scored
            else True,
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out = (
        args.out
        / f"sudoku__arch-{args.arch}__res-{args.cell_px}px__tau-{args.tau}__seed-{args.seed}__ceiling.csv"
    )
    table.to_csv(out, index=False)
    print(out, flush=True)


if __name__ == "__main__":
    main()
