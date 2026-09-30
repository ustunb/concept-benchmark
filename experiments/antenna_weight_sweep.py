"""Sweep the weight of a feature only the DNN can use: antennae in the balanced robot rule.

Rule for antenna weight w:
    s = 6*(mouth closed + body round + head round + ears triangle) + w*antennae + 8*pointy - 3*knees + c(w)
    P(Glorp) = sigma(4.2 * s)
c(w) makes the classes balanced over the full robot catalog. Antennae stay visible in the images but are
dropped from both concept sets, so the CBM cannot use or correct them. Everything after the dataset
(CBM, DNN, interventions, collect) is the committed pipeline.

Run from a code checkout whose data/robot_images already holds the rendered images:
    python experiments/antenna_weight_sweep.py --weight 6 --seed 2001 --preset ground_truth
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.formula import F, LabelFormula
from concept_benchmark.synthetic.robot.catalog import generate_robot_catalog
from scripts.robot_pipeline import run, setup_dataset

TEMPERATURE = 4.2
CONCEPTS_HIDDEN_FROM_CBM = ("has_antennae",)


def build_formula(antenna_weight: float, intercept: float) -> LabelFormula:
    score = (
        6 * F("mouth_type").closed
        + 6 * F("body_shape").round
        + 6 * F("head_shape").round
        + 6 * F("ears_shape").triangle
        + antenna_weight * F("has_antennae").true
        + 8 * F("foot_shape").pointy
        - 3 * F("has_knees").true
        + intercept
    )
    return LabelFormula(score=score, temperature=TEMPERATURE, stochastic=True)


def solve_balanced_intercept(config: RobotBenchmarkConfig, antenna_weight: float) -> float:
    """Intercept c with mean P(Glorp) = 0.5 over every robot in the catalog."""
    settings = config.to_dict()
    catalog, _ = generate_robot_catalog(
        concepts=settings["concepts"],
        additional_features=settings["additional_features"],
        draw=False,
    )
    base_score = catalog.apply(build_formula(antenna_weight, 0.0).score, axis=1).to_numpy(float)
    return brentq(lambda c: expit(TEMPERATURE * (base_score + c)).mean() - 0.5, -100.0, 100.0)


def check_rule(config: RobotBenchmarkConfig, antenna_weight: float, intercept: float) -> dict:
    """Read the rule back out of the saved dataset and check it is the one intended."""
    data = load(config.get_dataset_path())
    formula = data.meta["labeling_function"]
    catalog = data.meta["catalog_df"]
    score = catalog.apply(formula.score, axis=1).to_numpy(float)
    splits = [data.train, data.validation, data.test]
    y = np.concatenate([s.y for s in splits])
    rows = catalog.index.get_indexer(np.concatenate([s.meta["df_indices"] for s in splits]))
    s = score[rows]
    decided = s != 0
    concepts = list(data.train.concepts)
    check = {
        "antenna_weight": antenna_weight,
        "intercept": intercept,
        "formula": str(formula),
        "temperature": formula.temperature,
        "class_balance": float(y.mean()),
        "against_score_rate": float((y[decided] != (s[decided] > 0)).mean()),
        "concepts": concepts,
        "n_concepts": len(concepts),
    }
    assert formula.temperature == TEMPERATURE, check
    assert "has_antennae" not in concepts and "has_elbows" not in concepts, check
    assert 0.4 < check["class_balance"] < 0.6, check
    return check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, required=True, help="Antenna weight w in the rule.")
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

    intercept = solve_balanced_intercept(config, args.weight)
    config.label_formula = build_formula(args.weight, intercept)

    data = setup_dataset(config)          # committed split (skewed foot subtypes) + preset exclusions
    data.drop_concepts(list(CONCEPTS_HIDDEN_FROM_CBM))
    save(data, config.get_dataset_path(), overwrite=True)

    check = check_rule(config, args.weight, intercept)
    config.get_dataset_path().with_name("rule_check.json").write_text(json.dumps(check, indent=2))
    print("RULE CHECK", json.dumps({k: v for k, v in check.items() if k != "concepts"}), flush=True)
    if args.is_setup_only:
        return

    run(config, stages=["cbm", "dnn", "intervene", "collect"])


if __name__ == "__main__":
    main()
