"""Retrain ECBM on human_concepts with its concept terms switched off (training lambda_xc = lambda_cy = 0).

The image encoder then learns only the label; everything else is unchanged. Compare the image-only accuracy with
the default ECBM (scripts/paper/diagnose_incomplete_concepts.py). The dataset of the seed must exist (stage `setup`).

    PYTHONPATH=. python scripts/paper/train_ecbm_label_only.py --seed 1015
"""

from __future__ import annotations

import argparse

from concept_benchmark.config import RobotBenchmarkConfig
from scripts.robot_pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    config = RobotBenchmarkConfig(
        seed=args.seed, concept_preset="foot_subtypes", cbm_family="ecbm", ecbm_lambda_xc=0.0, ecbm_lambda_cy=0.0
    )
    config.rng_seed = args.seed
    run(config, stages=["cbm"])


if __name__ == "__main__":
    main()
