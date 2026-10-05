"""What interventions do, per architecture, intervener and budget (balanced rule, seeds 1014-1023).

Reads the installed intervention records (results/paper/robot/balanced_rule/intervention_records, written by
`scripts/robot_pipeline.py --dump-interventions DIR`: concepts intervened on, detector output, answers given, true
values, labels before/after). For true_concepts and human_concepts and the perfect / expert / LLM interveners it reports,
averaged over seeds:
  asked     share of test robots intervened on
  det       accuracy of the model's own detector on the concepts intervened on
  ans       accuracy of the answers on those concepts
  before    label accuracy on the intervened robots before / after the intervention
  fix/break share of intervened robots whose label goes wrong -> right / right -> wrong

    python scripts/paper/summarize_intervention_records.py [--by-concept]
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict

import numpy as np

from _common import BALANCED_TAG, INTERVENTION_RECORDS

ARCH_TAG = {"cbm": "arch-cbm", "cem": "arch-cem", "probcbm": "arch-probcbm__mode-joint", "ecbm": "arch-ecbm"}
SEEDS = tuple(range(1014, 1024))
ARCHS = ("cbm", "cem", "probcbm", "ecbm")
SETS = ("true", "human")
WHO = ("perfect", "expert", "llm")
SKIP = {("probcbm", "true", 1014)}  # record of the replaced model; the paper uses the retrained one


def load(arch: str, concepts: str, who: str, seed: int) -> dict[str, dict]:
    """Budget label ("1", "3", "max") -> record of one installed intervention cell."""
    out = {}
    for budget in ("1", "3", "max"):
        path = INTERVENTION_RECORDS / f"{BALANCED_TAG}__concepts-{concepts}__{ARCH_TAG[arch]}__isrc-{who}__budget-{budget}__seed-{seed}__records.npz"
        out[budget] = np.load(path)
    return out


def summarize(d) -> dict[str, float]:
    mask, y = d["mask"], d["y"].astype(int)
    sel = mask.any(axis=1)
    truth = d["C_true"].astype(int)
    det = (d["C_pred"] >= 0.5).astype(int)
    ans = (d["C_answer"] >= 0.5).astype(int)
    before, after = d["y_pred_before"].astype(int) == y, d["y_pred_after"].astype(int) == y
    return {
        "asked": 100 * sel.mean(),
        "det": 100 * (det[mask] == truth[mask]).mean(),
        "ans": 100 * (ans[mask] == truth[mask]).mean(),
        "before": 100 * before[sel].mean(),
        "after": 100 * after[sel].mean(),
        "fix": 100 * (~before[sel] & after[sel]).mean(),
        "break": 100 * (before[sel] & ~after[sel]).mean(),
        "acc_before": 100 * before.mean(),
        "acc_after": 100 * after.mean(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-concept", action="store_true", help="also list the asked concepts and per-concept accuracies at k=1")
    args = ap.parse_args()
    cols = ("asked", "det", "ans", "before", "after", "fix", "break", "acc_before", "acc_after")
    for concepts in SETS:
        print(f"\n=== {concepts}_concepts ===")
        print(f"{'':26s}" + "".join(f"{c:>11s}" for c in cols))
        for arch in ARCHS:
            for who in WHO:
                rows = defaultdict(list)
                per_concept = defaultdict(lambda: [0, 0, 0, 0])  # asked, det right, answer right, robots
                for seed in SEEDS:
                    if (arch, concepts, seed) in SKIP:
                        continue
                    dumps = load(arch, concepts, who, seed)
                    for k, d in dumps.items():
                        rows[k].append(summarize(d))
                    d = dumps["1"]
                    truth, mask = d["C_true"].astype(int), d["mask"]
                    for j, name in enumerate(d["concept_names"]):
                        m = mask[:, j]
                        pc = per_concept[str(name)]
                        pc[0] += int(m.sum())
                        pc[1] += int(((d["C_pred"][m, j] >= 0.5) == truth[m, j]).sum())
                        pc[2] += int(((d["C_answer"][m, j] >= 0.5) == truth[m, j]).sum())
                for k in ("1", "3", "max"):
                    print(f"{arch:8s}{who:8s} k={k:4s}   " + "".join(f"{st.mean(r[c] for r in rows[k]):11.1f}" for c in cols))
                if args.by_concept:
                    total = sum(v[0] for v in per_concept.values())
                    for name, (n, dr, ar, _) in sorted(per_concept.items(), key=lambda kv: -kv[1][0]):
                        if n / total >= 0.02:
                            print(f"{'':12s}k=1 asks {name:28s} {100 * n / total:5.1f}% of questions | detector {100 * dr / n:5.1f}% | answer {100 * ar / n:5.1f}%")


if __name__ == "__main__":
    main()
