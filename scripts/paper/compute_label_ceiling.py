"""Best achievable test accuracy under the balanced rule's stochastic labels.

Labels are drawn as y ~ Bernoulli(sigma(4.2 * score)), so even the rule itself is wrong on some robots. Two ceilings,
averaged over seeds:
  - rule:     knows every feature of each robot, predicts the likelier class: mean max(p, 1 - p);
  - concepts: knows only the true concepts (HasElbows is not one of them), predicts the likelier class within each
              combination of concept values: what a CBM with every concept intervened on can reach at best.

    python scripts/paper/compute_label_ceiling.py
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from concept_benchmark.ext.fileutils import load  # noqa: E402

DATASETS = REPO / "results/paper/robot/datasets"
SEEDS = (1014, 1015, 1016, 1017)


def ceilings(seed: int) -> tuple[float, float]:
    data = load(DATASETS / f"robot__rule-balanced__sampling-skew0.30__elbows-weight2__concepts-true__seed-{seed}__dataset.data")
    formula, catalog = data.meta["labeling_function"], data.meta["catalog_df"]
    test = catalog.iloc[catalog.index.get_indexer(data.test.meta["df_indices"])]
    p = expit(formula.temperature * test.apply(formula.score, axis=1).to_numpy(float))
    rule = float(np.maximum(p, 1 - p).mean())
    groups = pd.DataFrame(np.asarray(data.test.C)).astype(int).astype(str).agg("".join, axis=1)
    p_group = pd.Series(p).groupby(groups.values).transform("mean").to_numpy()
    concepts = float(np.maximum(p_group, 1 - p_group).mean())
    return rule, concepts


def main() -> None:
    values = [ceilings(seed) for seed in SEEDS]
    for seed, (rule, concepts) in zip(SEEDS, values):
        print(f"seed {seed}: rule {100 * rule:.1f}%, true concepts only {100 * concepts:.1f}%")
    print(f"mean: rule {100 * st.mean(v[0] for v in values):.1f}%, "
          f"true concepts only {100 * st.mean(v[1] for v in values):.1f}%")


if __name__ == "__main__":
    main()
