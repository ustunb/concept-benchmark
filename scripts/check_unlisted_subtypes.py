"""Why human_concepts cap the CBM in Figure 1a: robots whose foot subtype is not in the concept set.

For the Figure 1a runs (balanced rule, results/paper/robot/balanced_rule), sets every concept to its true value
(a full intervention, so the prediction is the CBM frontend on the true concepts) and reports accuracy separately
for test robots whose foot subtype is listed in `human_concepts` and those whose subtype is not. Also checks that
unlisted-subtype robots have every FootShape concept equal to 0.

Run from a code checkout of `grid-seeded-lfcbm` (models are pickled with that code):
    cd <checkout> && PYTHONPATH=. python <repo>/scripts/check_unlisted_subtypes.py
"""

from __future__ import annotations

import statistics as st
from pathlib import Path

import numpy as np

from concept_benchmark.ext.fileutils import load

PAPER = Path(__file__).resolve().parent.parent / "results/paper"
PREFIX = "robot__rule-balanced__sampling-skew0.30__elbows-weight2"
SEEDS = range(1014, 1024)


def allow_old_sklearn(obj, seen: set) -> None:
    """Binary LogisticRegression pickled with scikit-learn 1.8 lacks `multi_class`; older versions need it."""
    if id(obj) in seen or not hasattr(obj, "__dict__"):
        return
    seen.add(id(obj))
    if type(obj).__name__ == "LogisticRegression" and not hasattr(obj, "multi_class"):
        obj.multi_class = "auto"
    for value in vars(obj).values():
        if hasattr(value, "__dict__"):
            allow_old_sklearn(value, seen)


def accuracy_by_subtype(concepts: str, seed: int, listed_subtypes: set[str]) -> tuple[float, float, float]:
    data = load(PAPER / f"robot/datasets/{PREFIX}__concepts-{concepts}__seed-{seed}__dataset.data")
    model = load(PAPER / f"models/robot/balanced/{PREFIX}__concepts-{concepts}__arch-cbm__seed-{seed}__model.pt")
    allow_old_sklearn(model, set())
    test = data.test
    C = np.asarray(test.C).astype(int)
    correct = model.label_predictor.predict_proba(C).argmax(axis=1) == np.asarray(test.y).astype(int)
    catalog = data.meta["catalog_df"]
    rows = catalog.index.get_indexer(test.meta["df_indices"])
    subtype = (catalog["foot_shape"].astype(str) + "_" + catalog["foot_shape_subtype"].astype(str)).to_numpy()[rows]
    listed = np.isin(subtype, list(listed_subtypes))
    if concepts == "human":
        foot = [i for i, name in enumerate(test.concepts) if name.startswith("foot_shape_")]
        assert ((C[:, foot].sum(axis=1) == 0) == ~listed).all(), "unlisted robots should have no FootShape concept"
    return correct[listed].mean(), correct[~listed].mean(), (~listed).mean()


def main() -> None:
    human = load(PAPER / f"robot/datasets/{PREFIX}__concepts-human__seed-{SEEDS[0]}__dataset.data")
    listed = {c.removeprefix("foot_shape_") for c in human.train.concepts if c.startswith("foot_shape_")}
    print("FootShape subtypes in human_concepts:", ", ".join(sorted(listed)))
    for concepts in ("true", "human"):
        on_listed, on_unlisted, share = zip(*(accuracy_by_subtype(concepts, s, listed) for s in SEEDS))
        ms = lambda v: f"{100 * st.mean(v):.1f} ± {100 * st.stdev(v) / len(v) ** 0.5:.1f}"
        print(f"{concepts:5} concepts, full intervention: listed-subtype robots {ms(on_listed)} | "
              f"unlisted-subtype robots {ms(on_unlisted)} (unlisted share {100 * st.mean(share):.1f}%)")


if __name__ == "__main__":
    main()
