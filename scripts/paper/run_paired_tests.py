"""Paired tests behind the findings of the architecture and automation sections (balanced rule, seeds 1014-1023).

Every test pairs on the seed (same data, same split). Reads the installed cells in results/paper/robot/balanced_rule;
label-free CBM cells use the 5th/95th-percentile interventions.

    python scripts/paper/run_paired_tests.py
"""

from __future__ import annotations

import csv
import statistics as st

from scipy import stats

from _common import BALANCED_RULE, BALANCED_TAG, mean_se, read_budget_rows

SEEDS = tuple(range(1014, 1024))
ARCHS = {"cbm": "arch-cbm", "cem": "arch-cem", "probcbm": "arch-probcbm__mode-joint", "ecbm": "arch-ecbm"}
AUTOMATED = ("machine", "llm", "clip")
BUDGETS = ("k=0", "k=1", "k=3", "k=max")


def accuracy(arch: str, concepts: str, isrc: str) -> list[list[float]]:
    """Per-seed accuracies (%) at k = 0, 1, 3, max."""
    out = []
    for seed in SEEDS:
        files = [f for f in BALANCED_RULE.glob(f"{BALANCED_TAG}__concepts-{concepts}__{ARCHS[arch]}__isrc-{isrc}*__seed-{seed}__results.csv")
                 if (isrc != "llm" or "img-224px" in f.name)
                 and (("enc-koh595" in f.name) == (arch == "cbm" and concepts in AUTOMATED) or "enc-" not in f.name)]
        if len(files) != 1:
            raise SystemExit(f"{arch} {concepts} {isrc} seed {seed}: {[f.name for f in files]}")
        rows = read_budget_rows(files[0])
        out.append([100 * float(r["accuracy"]) for r in rows])
    return out


def dnn() -> list[float]:
    out = []
    for seed in SEEDS:
        path = BALANCED_RULE / f"{BALANCED_TAG}__concepts-human__arch-cbm-and-dnn__seed-{seed}__results.csv"
        out += [100 * float(r["accuracy"]) for r in csv.DictReader(path.open()) if "dnn" in r["model"].lower()]
    return out


def paired(name: str, a: list[float], b: list[float]) -> None:
    diff = [x - y for x, y in zip(a, b)]
    t, p = stats.ttest_rel(a, b)
    print(f"  {name:58s} diff {st.mean(diff):+6.2f}  t={t:+6.2f}  p={p:.4f}  seeds with diff>0: {sum(d > 0 for d in diff)}/{len(diff)}")


def gain(runs: list[list[float]]) -> list[float]:
    return [st.mean(r[1:]) - r[0] for r in runs]


def main() -> None:
    d = dnn()
    print("DNN {:.2f} ± {:.2f}".format(*mean_se(d)))

    print("\nArchitectures on human_concepts, perfect interventions: architecture − DNN")
    human = {a: accuracy(a, "human", "perfect") for a in ARCHS}
    for a, runs in human.items():
        print(f" {a}: mean over budgets {st.mean(st.mean(r) for r in runs):.2f}")
        for k, name in enumerate(BUDGETS):
            paired(f"{a} {name} − DNN", [r[k] for r in runs], d)

    print("\nProbCBM on human_concepts, perfect interventions: change from k=0")
    for k in (1, 2, 3):
        paired(f"probcbm {BUDGETS[k]} − k=0", [r[k] for r in human["probcbm"]], [r[0] for r in human["probcbm"]])

    print("\nAutomated concept sets, perfect interventions: mean accuracy after interventions − k=0")
    pooled = {a: [0.0] * len(SEEDS) for a in ARCHS}
    for a in ARCHS:
        for c in AUTOMATED:
            runs = accuracy(a, c, "perfect")
            paired(f"{a} {c}", [st.mean(r[1:]) for r in runs], [r[0] for r in runs])
            pooled[a] = [x + g / len(AUTOMATED) for x, g in zip(pooled[a], gain(runs))]
    for a in ARCHS:
        paired(f"{a}, mean over the three concept sets (vs 0)", pooled[a], [0.0] * len(SEEDS))
    three = [st.mean(pooled[a][i] for a in ("cbm", "probcbm", "ecbm")) for i in range(len(SEEDS))]
    paired("CBM, ProbCBM, ECBM pooled (vs 0)", three, [0.0] * len(SEEDS))

    print("\nLLM vs expert interventions on true/human concepts: gain (mean after − k=0)")
    for c in ("true", "human"):
        for a in ARCHS:
            llm, expert = gain(accuracy(a, c, "llm")), gain(accuracy(a, c, "expert"))
            paired(f"{a} {c}: LLM gain (vs 0)", llm, [0.0] * len(SEEDS))
            paired(f"{a} {c}: expert gain (vs 0)", expert, [0.0] * len(SEEDS))
            paired(f"{a} {c}: LLM gain − expert gain", llm, expert)

    print("\nCBM on hand-annotated concepts (mean of true and human): share of the perfect-intervention gain recovered")
    g = {w: [st.mean(x) for x in zip(gain(accuracy("cbm", "true", w)), gain(accuracy("cbm", "human", w)))]
         for w in ("perfect", "expert", "llm")}
    share = {w: [100 * a / b for a, b in zip(g[w], g["perfect"])] for w in ("expert", "llm")}
    print(f"  gains: perfect {st.mean(g['perfect']):+.2f}, expert {st.mean(g['expert']):+.2f}, LLM {st.mean(g['llm']):+.2f}")
    paired("LLM share − expert share (per seed, %)", share["llm"], share["expert"])
    print(f"  mean share: expert {st.mean(share['expert']):.1f}%, LLM {st.mean(share['llm']):.1f}%")


if __name__ == "__main__":
    main()
