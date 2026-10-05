"""Retrain ECBM on human_concepts with its concept terms switched off (training lambda_xc = lambda_cy = 0).

Runs the section 3 training command (skew_sweep, balanced rule, cbm stage only) with the two ECBM training weights
set to zero on the config, so the image encoder learns only the label. Everything else is unchanged. Compare the
image-only accuracy with the installed ECBM (scripts/paper/diagnose_incomplete_concepts.py).

    cd <run dir> && PYTHONPATH=. python scripts/paper/train_ecbm_label_only.py --seed 1015
"""

from __future__ import annotations

import argparse
import sys

import scripts.robot_pipeline as pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    original_run = pipeline.run

    def run_label_only(config, stages=None):
        config.ecbm_lambda_xc = 0.0
        config.ecbm_lambda_cy = 0.0
        print("ECBM training weights xy/xc/cy:", config.ecbm_lambda_xy, config.ecbm_lambda_xc, config.ecbm_lambda_cy, flush=True)
        return original_run(config, stages=stages)

    import experiments.skew_sweep as sweep
    sweep.run = run_label_only
    sys.argv = ["skew_sweep.py", "--dominant-fraction", "0.30", "--elbows-weight", "2", "--seed", str(args.seed),
                "--skip-setup", "--preset", "foot_subtypes", "--", "--cbm-family", "ecbm", "--stages", "cbm"]
    sweep.main()


if __name__ == "__main__":
    main()
