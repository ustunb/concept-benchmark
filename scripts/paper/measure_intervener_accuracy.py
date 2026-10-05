"""How accurate is each intervener on the concepts that actually get intervened on? (balanced rule, CBM)

Reruns the pipeline's KFlip policy (k = 1, threshold 0.2) on the installed CBM of each seed to find the (robot,
concept) pairs it selects, then compares on exactly those pairs: the CBM's own estimate (probability >= 0.5), the
Gemini answer from the installed LLM cache, and the simulated expert (80% accurate on every concept by construction).
Also reports how far the CBM's probability is from 0.5 there and which concepts are selected.

    PYTHONPATH=. python scripts/paper/measure_intervener_accuracy.py [--concepts human] [--budget 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path[:0] = [str(REPO / "scripts" / "paper"), str(REPO / "scripts")]
from concept_benchmark.ext.fileutils import load  # noqa: E402
from experiments.intervention import InterventionBatch, InterventionConfig  # noqa: E402
from experiments.kflip import KFlipInterventionStrategy  # noqa: E402
from measure_intervention_response import set_cpu  # noqa: E402

PAPER = REPO / "results/paper"
TAG = "robot__rule-balanced__sampling-skew0.30__elbows-weight2"
LLM = "llm-gemini-2.5-flash-lite__img-224px__questions-v2-value-explicit"
IMAGES = REPO / "data/robot_images"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", choices=["true", "human"], default="true")
    ap.add_argument("--budget", type=int, default=1)
    args = ap.parse_args()
    model_tag, cache_tag = {"true": ("ideal", "true"), "human": ("subconcept", "human-and-machine")}[args.concepts]
    for seed in (1014, 1015, 1016, 1017):
        test = load(PAPER / f"robot/datasets/{TAG}__concepts-{args.concepts}__seed-{seed}__dataset.data").test
        test.base_dir = IMAGES
        model = load(next((PAPER / "models/robot/balanced").glob(f"{TAG}__{model_tag}_cbm_seed{seed}__*")))
        set_cpu(model)
        P = model.concept_detector.predict_proba(test)
        C = np.asarray(test.C).astype(int)
        # as in the pipeline: at k = max every concept of a selected robot is corrected
        proposal = KFlipInterventionStrategy(use_exact_k=args.budget >= C.shape[1]).propose(
            model, InterventionBatch(C_pred=P, C_true=C, y_true=np.asarray(test.y)),
            InterventionConfig(per_instance_budget=args.budget, random_state=seed, score_threshold=0.2))
        mask = np.asarray(proposal.mask, dtype=bool)
        votes = np.full(C.shape, -1)
        for line in open(PAPER / f"robot/llm_caches/robot__concepts-{cache_tag}__{LLM}__seed-{seed}__llm-votes.jsonl"):
            r = json.loads(line)
            votes[int(r["i"])] = [r["votes_idx"][str(j)] for j in range(C.shape[1])]
        own = (P >= 0.5).astype(int)
        names = list(test.concepts)
        sel = Counter(names[j] for j in np.nonzero(mask)[1])
        print(f"seed {seed}: {mask.sum()} intervened pairs on {mask.any(axis=1).sum()} robots | "
              f"LLM acc on them {np.mean(votes[mask] == C[mask]):.3f} (overall {np.mean(votes == C):.3f}) | "
              f"CBM estimate acc on them {np.mean(own[mask] == C[mask]):.3f} | expert 0.800 | "
              f"mean |p-0.5| there {np.mean(np.abs(P[mask] - 0.5)):.2f} | concepts {dict(sel.most_common())}", flush=True)
        right = own[mask] == C[mask]
        llm_right = votes[mask] == C[mask]
        print(f"    CBM estimate right on {right.mean():.3f} of picked pairs; LLM breaks {np.mean(~llm_right[right]):.3f} of the right "
              f"ones and fixes {np.mean(llm_right[~right]):.3f} of the wrong ones", flush=True)
        for j, n in enumerate(names):
            m = mask[:, j]
            if m.sum():
                print(f"    {n:13s} picked {m.sum():5d}  LLM acc {np.mean(votes[m, j] == C[m, j]):.3f}  "
                      f"CBM acc {np.mean(own[m, j] == C[m, j]):.3f}", flush=True)


if __name__ == "__main__":
    main()
