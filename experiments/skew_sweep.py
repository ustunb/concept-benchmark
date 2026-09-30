"""Sweep the foot-subtype skew of the training set under the balanced robot rule.

Rule (fixed, as in the appendix):
    s = 6*(mouth closed + body round + head round + antennae + ears triangle) + 8*pointy - 3*knees - 3*elbows - 16
    P(Glorp) = sigma(4.2 * s)
HasElbows drives the label but is not a concept. Only the training skew over foot subtypes changes; the test set
stays uniform. The split itself is the committed `setup_dataset`, given a different constraint list.

Run from a code checkout whose data/robot_images already holds the rendered images:
    python experiments/skew_sweep.py --level s40 --seed 2001 --preset ground_truth
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import concept_benchmark.config as benchmark_config
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load
from concept_benchmark.formula import F, LabelFormula
from scripts.robot_pipeline import run, setup_dataset

TEMPERATURE = 4.2
DOMINANT = ("foot_shape_pointy_4sided", "foot_shape_flat_5sided")
RARE = ("foot_shape_pointy_square", "foot_shape_pointy_rounded", "foot_shape_flat_square", "foot_shape_flat_trapezoid")
DOMINANT_FRACTION = {"s49": 0.49, "s40": 0.40, "s30": 0.30}


def skew_constraints(level: str) -> list[dict]:
    """Training-set constraints: two dominant subtypes share `DOMINANT_FRACTION`, the four rare ones split the rest."""
    if level == "uniform":
        return []
    dominant = DOMINANT_FRACTION[level]
    rare = (1.0 - 2 * dominant) / len(RARE)
    return [{"concepts": {name: 1}, "min_fraction": dominant} for name in DOMINANT] + [
        {"concepts": {name: 1}, "min_fraction": rare} for name in RARE
    ]


def build_formula() -> LabelFormula:
    score = (
        6 * F("mouth_type").closed
        + 6 * F("body_shape").round
        + 6 * F("head_shape").round
        + 6 * F("has_antennae").true
        + 6 * F("ears_shape").triangle
        + 8 * F("foot_shape").pointy
        - 3 * F("has_knees").true
        - 3 * F("has_elbows").true
        - 16
    )
    return LabelFormula(score=score, temperature=TEMPERATURE, stochastic=True)


def train_subtype_fractions(config: RobotBenchmarkConfig) -> dict[str, float]:
    data = load(config.get_dataset_path())
    catalog = data.meta["catalog_df"]
    rows = catalog.index.get_indexer(data.train.meta["df_indices"])
    subtype = catalog["foot_shape"].astype(str) + "_" + catalog["foot_shape_subtype"].astype(str)
    return {k: round(float(v), 4) for k, v in subtype.iloc[rows].value_counts(normalize=True).items()}


def check_rule(config: RobotBenchmarkConfig, level: str) -> dict:
    """Read the rule and the training skew back out of the saved dataset."""
    data = load(config.get_dataset_path())
    formula = data.meta["labeling_function"]
    catalog = data.meta["catalog_df"]
    score = catalog.apply(formula.score, axis=1).to_numpy(float)
    splits = [data.train, data.validation, data.test]
    y = np.concatenate([s.y for s in splits])
    s = score[catalog.index.get_indexer(np.concatenate([sp.meta["df_indices"] for sp in splits]))]
    decided = s != 0
    fractions = train_subtype_fractions(config)
    check = {
        "level": level,
        "formula": str(formula),
        "temperature": formula.temperature,
        "class_balance": float(y.mean()),
        "against_score_rate": float((y[decided] != (s[decided] > 0)).mean()),
        "concepts": list(data.train.concepts),
        "train_subtype_fractions": fractions,
    }
    assert formula.temperature == TEMPERATURE, check
    assert "has_elbows" not in check["concepts"], check
    if level != "uniform":
        for name, target in [(n, DOMINANT_FRACTION[level]) for n in DOMINANT]:
            got = fractions.get(name.removeprefix("foot_shape_"), 0.0)
            assert abs(got - target) < 0.01, (name, got, target)
    return check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=[*DOMINANT_FRACTION, "uniform"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--preset", choices=["ground_truth", "foot_subtypes"], required=True)
    ap.add_argument("--is-setup-only", action="store_true", help="Build and check the dataset, then stop.")
    args = ap.parse_args()

    if args.preset == "foot_subtypes":
        config = RobotBenchmarkConfig.default_subconcept()
        config.seed = args.seed
    else:
        config = RobotBenchmarkConfig(seed=args.seed)
    config.rng_seed = args.seed           # labels drawn with the run seed
    config.render_images = False          # images are pre-rendered and shared between runs
    config.cbm_family = "cbm"
    config.intervention_budgets = [1, 3, -1]
    config.force_retrain = True
    config.label_formula = build_formula()

    # setup_dataset imports the constraint list at call time, so this swaps only the training skew
    benchmark_config.ROBOT_SAMPLING_CONSTRAINTS = skew_constraints(args.level)
    setup_dataset(config)

    check = check_rule(config, args.level)
    config.get_dataset_path().with_name("rule_check.json").write_text(json.dumps(check, indent=2))
    print("RULE CHECK", json.dumps({k: v for k, v in check.items() if k != "concepts"}), flush=True)
    if args.is_setup_only:
        return

    run(config, stages=["cbm", "dnn", "intervene", "collect"])


if __name__ == "__main__":
    main()
