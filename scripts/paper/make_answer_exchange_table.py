"""Write the answer-exchange table: which concept separates two interveners (CBM, balanced rule, seeds 1014-1023).

Reads the installed intervention records of two interveners (default: the LLM and the simulated expert) for the CBM on
true and human concepts. Both runs select the same robots and concepts, so their answers can be exchanged: the script
replays the CBM's label predictor on (a) each intervener's own answers, (b) the first intervener's answers with one
concept answered by the second, and (c) the reverse, and reports test accuracy at k = 1, 3, max. The replay must
reproduce the predictions saved by the pipeline exactly before anything is written.

    PYTHONPATH=. python scripts/paper/make_answer_exchange_table.py --out ../concept-benchmark-paper/tables/answer_exchange.tex
    PYTHONPATH=. python scripts/paper/make_answer_exchange_table.py --concept foot_shape --concept-sets true   # print only
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from concept_benchmark.ext.fileutils import load  # noqa: E402

PAPER = REPO / "results/paper"
TAG = "robot__rule-balanced__sampling-skew0.30__elbows-weight2"
RECORDS = PAPER / "robot/balanced_rule/intervention_records"
SEEDS = tuple(range(1014, 1024))
MODEL_TAG = {"true": "ideal", "human": "subconcept"}
BUDGETS = ("1", "3", "max")
NAMES = {"llm": "LLM", "expert": "Expert", "perfect": "True values"}
CONCEPT_MACRO = {"has_knees": "HasKnees", "foot_shape": "FootShape", "head_shape": "HeadShape", "body_shape": "BodyShape",
                 "has_antennae": "HasAntennae", "ears_shape": "EarShape", "mouth_type": "MouthType"}


def record(concepts: str, isrc: str, budget: str, seed: int):
    return np.load(RECORDS / f"{TAG}__concepts-{concepts}__arch-cbm__isrc-{isrc}__budget-{budget}__seed-{seed}__records.npz")


def label_predictor(concepts: str, seed: int):
    """The CBM's label predictor as a function of concept values (it reads thresholded concepts)."""
    model = load(next((PAPER / "models/robot/balanced").glob(f"{TAG}__{MODEL_TAG[concepts]}_cbm_seed{seed}__*")))
    logistic = model.label_predictor.model
    weights, bias, classes = logistic.coef_.ravel(), float(logistic.intercept_.ravel()[0]), logistic.classes_
    return lambda C: classes[((np.asarray(C) >= 0.5).astype(float) @ weights + bias > 0).astype(int)]


def exchange(concepts: str, first: str, second: str, concept: str) -> dict[str, list[float]]:
    """Mean test accuracy (percent) at k = 1, 3, max for each row of the table, and before interventions."""
    rows: dict[str, list[list[float]]] = {}
    for seed in SEEDS:
        predict = label_predictor(concepts, seed)
        for b, budget in enumerate(BUDGETS):
            a, c = record(concepts, first, budget, seed), record(concepts, second, budget, seed)
            if not np.array_equal(a["mask"], c["mask"]):
                raise SystemExit(f"{concepts} seed {seed} k={budget}: the two interveners were asked about different concepts")
            for r in (a, c):
                if not np.array_equal(predict(r["C_answer"]), r["y_pred_after"]):
                    raise SystemExit(f"{concepts} seed {seed} k={budget}: the replay does not reproduce the saved predictions")
            y = a["y"].astype(int)
            j = list(map(str, a["concept_names"])).index(concept)
            first_with_second, second_with_first = a["C_answer"].copy(), c["C_answer"].copy()
            first_with_second[:, j], second_with_first[:, j] = c["C_answer"][:, j], a["C_answer"][:, j]
            for name, answers in (("before", a["C_pred"]), (second, c["C_answer"]), (first, a["C_answer"]),
                                  (f"{first}+{second}", first_with_second), (f"{second}+{first}", second_with_first)):
                rows.setdefault(name, [[] for _ in BUDGETS])[b].append(100 * float((predict(answers) == y).mean()))
    return {name: [st.mean(v) for v in per_budget] for name, per_budget in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="llm", choices=sorted(NAMES))
    ap.add_argument("--second", default="expert", choices=sorted(NAMES))
    ap.add_argument("--concept", default="has_knees", help="concept whose answers are exchanged")
    ap.add_argument("--concept-sets", nargs="+", default=["true", "human"], choices=sorted(MODEL_TAG))
    ap.add_argument("--out", type=Path, default=None, help="write the LaTeX table here (default: print only)")
    args = ap.parse_args()
    first, second = NAMES[args.first], NAMES[args.second]
    results = {c: exchange(c, args.first, args.second, args.concept) for c in args.concept_sets}
    labels = [(args.second, f"{second} on every concept"), (args.first, f"{first} on every concept"),
              (f"{args.first}+{args.second}", f"{first}, with the {second.lower()}'s answers on \\textcp{{{CONCEPT_MACRO.get(args.concept, args.concept)}}}"),
              (f"{args.second}+{args.first}", f"{second}, with the {first}'s answers on \\textcp{{{CONCEPT_MACRO.get(args.concept, args.concept)}}}")]
    for c, rows in results.items():
        print(f"{c}_concepts (before interventions {rows['before'][0]:.1f}%); k = 1 / 3 / max")
        for key, label in labels:
            print(f"   {label:70s} " + " / ".join(f"{v:.1f}" for v in rows[key]))
    if args.out is None:
        return
    n = len(args.concept_sets)
    lines = [rf"\begin{{tabular}}{{@{{}}l{'rrr' * n}@{{}}}}", r"\toprule",
             "& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textds{{{c}\_concepts}}}}" for c in args.concept_sets) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n)),
             r"\textheader{Answers} & " + " & ".join([r"$k{=}1$ & $k{=}3$ & $k{=}\text{max}$"] * n) + r" \\", r"\midrule"]
    for i, (key, label) in enumerate(labels):
        if i == 2:
            lines.append(r"\midrule")
        lines.append(label + " & " + " & ".join(f"{v:.1f}" for c in args.concept_sets for v in results[c][key]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
