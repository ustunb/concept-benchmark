"""How many concepts per robot is each model unsure about (0.01 < P < 0.99)? Decides the cost of up-to-max KFlip.

Runs the section 3 intervene command for one architecture with every concept source, but replaces each cell by a count:
for every test robot, the number of concepts whose predicted probability lies strictly between 0.01 and 0.99, and the
number of KFlip assignments up to max over those concepts (sum over subsets S of 2^|S| = 3^u - 1), against today's up
to 3 over all concepts.

    cd <run dir> && PYTHONPATH=. python scripts/paper/count_uncertain_concepts.py --seed 1014 --family cem
"""

from __future__ import annotations

import argparse
import sys
from math import comb

import numpy as np
import pandas as pd

import scripts.robot_pipeline as pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--family", required=True)
    args = ap.parse_args()

    def count_cell(config, concept_source, intervention_source, family, data, budgets, thresholds):
        _, c_preds, names, _ = pipeline._prepare_model_for_concept_source(config, concept_source, family, data)
        P = np.asarray(c_preds, dtype=float)
        u = ((P > 0.01) & (P < 0.99)).sum(axis=1)
        n = P.shape[1]
        up_to_3_all = sum(comb(n, s) * 2 ** s for s in range(1, 4))
        pruned_max = float(np.mean(3.0 ** u - 1))
        print(f"COUNT family={family} source={concept_source} concepts={n} uncertain per robot: mean {u.mean():.2f} "
              f"median {np.median(u):.0f} p90 {np.percentile(u, 90):.0f} max {u.max()} | assignments per robot: "
              f"up-to-max pruned {pruned_max:.0f} vs up-to-3 today {up_to_3_all}", flush=True)
        return pd.DataFrame({"budget": [0], "accuracy": [0.0], "concept_source": [concept_source],
                             "intervention_source": [intervention_source]})

    pipeline._run_cell = count_cell
    import experiments.skew_sweep as sweep
    for preset, sources in (("ground_truth", ["ground_truth"]),
                            ("foot_subtypes", ["human_concepts", "machine_annotation", "llm_concepts", "clip_concepts"])):
        sys.argv = ["skew_sweep.py", "--dominant-fraction", "0.30", "--elbows-weight", "2", "--seed", str(args.seed),
                    "--skip-setup", "--preset", preset, "--", "--cbm-family", args.family, "--budgets", "1",
                    *(["--training-mode", "joint"] if args.family == "probcbm" else []),
                    "--concept-sources", *sources, "--intervention-sources", "perfect", "--stages", "intervene"]
        sweep.main()


if __name__ == "__main__":
    main()
