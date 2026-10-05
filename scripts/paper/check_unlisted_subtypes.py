"""Why human_concepts cap the CBM in Figure 1a: robots whose foot subtype is not in the concept set.

For the Figure 1a runs (balanced rule, results/paper/robot/balanced_rule), sets every concept to its true value
(a full intervention, so the prediction is the CBM frontend on the true concepts) and reports accuracy separately
for test robots whose foot subtype is listed in `human_concepts` and those whose subtype is not. Also checks that
unlisted-subtype robots have every FootShape concept equal to 0.

With --images, also runs the human-concept CBM's concept detector on the test robots and reports, for robots with
an unlisted subtype, how often any FootShape detector fires (a wrong prediction, since every true value is 0) and
how often the firing subtype is on the correct Pointy/Flat side, plus accuracy before interventions.

Run with the code version that trained the models (they are pickled with it):
    cd <checkout> && PYTHONPATH=. python <repo>/scripts/paper/check_unlisted_subtypes.py [--images <robot_images>]
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

from _common import BALANCED_TAG, PAPER_RESULTS, ROBOT_DATASETS
from concept_benchmark.ext.fileutils import load

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


def accuracy_by_subtype(
    concepts: str, seed: int, listed_subtypes: set[str]
) -> tuple[float, float, float]:
    data = load(
        ROBOT_DATASETS
        / f"{BALANCED_TAG}__concepts-{concepts}__seed-{seed}__dataset.data"
    )
    model = load(
        PAPER_RESULTS
        / f"models/robot/balanced/{BALANCED_TAG}__concepts-{concepts}__arch-cbm__seed-{seed}__model.pt"
    )
    allow_old_sklearn(model, set())
    test = data.test
    C = np.asarray(test.C).astype(int)
    correct = model.label_predictor.predict_proba(C).argmax(axis=1) == np.asarray(
        test.y
    ).astype(int)
    catalog = data.meta["catalog_df"]
    rows = catalog.index.get_indexer(test.meta["df_indices"])
    subtype = (
        catalog["foot_shape"].astype(str)
        + "_"
        + catalog["foot_shape_subtype"].astype(str)
    ).to_numpy()[rows]
    listed = np.isin(subtype, list(listed_subtypes))
    if concepts == "human":
        foot = [
            i for i, name in enumerate(test.concepts) if name.startswith("foot_shape_")
        ]
        assert ((C[:, foot].sum(axis=1) == 0) == ~listed).all(), (
            "unlisted robots should have no FootShape concept"
        )
    return correct[listed].mean(), correct[~listed].mean(), (~listed).mean()


def detector_on_unlisted(
    seed: int, listed_subtypes: set[str], images: Path
) -> tuple[float, float, float, float]:
    """Share of unlisted-subtype robots with any FootShape detector firing, share of those firing on the correct
    Pointy/Flat side, and k=0 accuracy on unlisted vs listed robots (human_concepts CBM)."""
    data = load(
        ROBOT_DATASETS / f"{BALANCED_TAG}__concepts-human__seed-{seed}__dataset.data"
    )
    model = load(
        PAPER_RESULTS
        / f"models/robot/balanced/{BALANCED_TAG}__concepts-human__arch-cbm__seed-{seed}__model.pt"
    )
    allow_old_sklearn(model, set())
    for cfg_name in ("_eval_config", "eval_config"):
        cfg = getattr(model.concept_detector, cfg_name, None)
        if isinstance(cfg, dict):
            cfg["device"] = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    test = data.test
    test.base_dir = Path(images)
    probs = model.concept_detector.predict_proba(test)
    names = list(test.concepts)
    foot = [i for i, n in enumerate(names) if n.startswith("foot_shape_")]
    fired = probs[:, foot] >= 0.5
    catalog = data.meta["catalog_df"]
    rows = catalog.index.get_indexer(test.meta["df_indices"])
    side = catalog["foot_shape"].astype(str).to_numpy()[rows]
    subtype = (
        catalog["foot_shape"].astype(str)
        + "_"
        + catalog["foot_shape_subtype"].astype(str)
    ).to_numpy()[rows]
    unlisted = ~np.isin(subtype, list(listed_subtypes))
    any_fired = fired.any(axis=1)
    fired_side = np.array(
        [names[foot[j]].split("_")[2] for j in range(len(foot))]
    )  # pointy/flat per concept
    best = probs[:, foot].argmax(axis=1)
    same_side = fired_side[best] == side
    pred = model.label_predictor.predict_proba((probs >= 0.5).astype(int)).argmax(
        axis=1
    )
    correct = pred == np.asarray(test.y).astype(int)
    return (
        any_fired[unlisted].mean(),
        same_side[unlisted & any_fired].mean(),
        correct[unlisted].mean(),
        correct[~unlisted].mean(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Robot images; enables the detector check.",
    )
    args = ap.parse_args()
    human = load(
        ROBOT_DATASETS
        / f"{BALANCED_TAG}__concepts-human__seed-{SEEDS[0]}__dataset.data"
    )
    listed = {
        c.removeprefix("foot_shape_")
        for c in human.train.concepts
        if c.startswith("foot_shape_")
    }
    print("FootShape subtypes in human_concepts:", ", ".join(sorted(listed)))
    ms = lambda v: f"{100 * st.mean(v):.1f} ± {100 * st.stdev(v) / len(v) ** 0.5:.1f}"
    for concepts in ("true", "human"):
        on_listed, on_unlisted, share = zip(
            *(accuracy_by_subtype(concepts, s, listed) for s in SEEDS)
        )
        print(
            f"{concepts:5} concepts, full intervention: listed-subtype robots {ms(on_listed)} | "
            f"unlisted-subtype robots {ms(on_unlisted)} (unlisted share {100 * st.mean(share):.1f}%)"
        )
    if args.images is not None:
        fired, same_side, k0_unlisted, k0_listed = zip(
            *(detector_on_unlisted(s, listed, args.images) for s in SEEDS)
        )
        print(
            f"human concepts, detectors on unlisted-subtype robots: some FootShape detector fires on {ms(fired)}% "
            f"(true value 0); of those, the most confident fires on the correct Pointy/Flat side in {ms(same_side)}%"
        )
        print(
            f"human concepts, no intervention: accuracy {ms(k0_unlisted)} on unlisted vs {ms(k0_listed)} on listed robots"
        )


if __name__ == "__main__":
    main()
