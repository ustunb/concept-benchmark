"""Make candidate A the cited Figure 1a data and demote B-uniform to the configuration search.

Candidate A (experiments/skew_sweep.py --dominant-fraction 0.30 --elbows-weight 2, HasElbows not a concept,
commit 898b0e41): the training sample holds only the six FootShape subtypes listed in human_concepts, so an
annotator who labels the subtypes they see would produce exactly that concept set; the test set holds all ten.
B-uniform trains on all ten subtypes, so a six-subtype concept set would not be what an annotator produces.

Moves B-uniform's cited files from robot/balanced_rule to the exploratory search folder (its models and datasets
stay in place, re-indexed as exploratory), installs A's results, rule checks, logs, models and datasets under
content names, removes the now-duplicate exploratory copy of A, and updates INDEX.csv.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "results/paper"
INCOMING = REPO / "results/_incoming/final_d030e2"
A_TAG = "d0.30e2"
A_PREFIX = "robot__rule-balanced__sampling-skew0.30__elbows-weight2"
B_PREFIX = "robot__rule-balanced__sampling-uniform__elbows-concept"
SEARCH = PAPER / "robot/exploratory/balanced-rule-search"
B_FOLDER = SEARCH / "candidate-B-uniform_elbows-concept_uniform_seeds-1014-1023"
A_DUPLICATE = SEARCH / "candidate-A_elbows-hidden_skew-0.30_seeds-1014-1023"
PRESETS = {"ground_truth": ("ideal", "true"), "foot_subtypes": ("subconcept", "human")}
B_NOTE = ("candidate B-uniform (not cited): training covers all ten subtypes, so a six-subtype human_concepts "
          "set is not what an annotator would label; superseded by candidate A")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    index_path = PAPER / "INDEX.csv"
    with index_path.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        by_path = {r["path"]: r for r in reader}

    def rel(p: Path) -> str:
        return str(p.relative_to(PAPER))

    # 1. B-uniform: cited files -> search folder; models and datasets re-indexed in place
    balanced = PAPER / "robot/balanced_rule"
    for f in sorted(balanced.rglob(f"{B_PREFIX}__*")):
        dest = B_FOLDER / f.relative_to(balanced)
        dest.parent.mkdir(parents=True, exist_ok=True)
        entry = by_path.pop(rel(f))
        shutil.move(f, dest)
        by_path[rel(dest)] = {**entry, "path": rel(dest), "role": "exploratory", "note": B_NOTE}
    for f in [*(PAPER / "models/robot/balanced").glob(f"{B_PREFIX}__*"), *(PAPER / "robot/datasets").glob(f"{B_PREFIX}__*")]:
        by_path[rel(f)] = {**by_path[rel(f)], "role": "exploratory", "note": B_NOTE}

    # 2. A: install as the cited Figure 1a data
    def place(source: Path, dest: Path, role: str, note: str) -> None:
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        by_path[rel(dest)] = {"path": rel(dest), "role": role, "sha256": sha256(dest),
                              "source": str(source.relative_to(REPO / "results")), "note": note}

    note = ("Figure 1a: balanced rule with elbows weight 2 (HasElbows not a concept), training sample = the six "
            "listed FootShape subtypes (0.30/0.30/0.10 x4), test uniform over all ten")
    for seed in range(1014, 1024):
        for preset, (variant, name) in PRESETS.items():
            run = INCOMING / f"run_{A_TAG}_s{seed}_{preset}"
            res = run / "results"
            if not (run / "DONE").exists():
                raise SystemExit(f"run not finished: {run}")
            prefix = f"{A_PREFIX}__concepts-{name}"
            place(next(res.glob(f"robot_{variant}_seed{seed}_*_results.csv")),
                  balanced / f"{prefix}__arch-cbm-and-dnn__seed-{seed}__results.csv", "cited", note)
            place(res / f"robot_image_stochastic_{variant}_cbm_seed{seed}_results.csv",
                  balanced / f"{prefix}__arch-cbm__isrc-perfect__strategy-upto__seed-{seed}__results.csv", "cited", note)
            place(res / "rule_check.json", balanced / f"{prefix}__seed-{seed}__rulecheck.json", "cited",
                  "labeling rule, class balance and training fractions read back from the dataset")
            place(run / "log.txt", balanced / "run_logs" / f"{prefix}__seed-{seed}__log.txt", "log", "run log")
            for arch in ("cbm", "dnn"):
                place(res / f"robot_image_stochastic_4_{variant}_{arch}_seed{seed}.model",
                      PAPER / "models/robot/balanced" / f"{prefix}__arch-{arch}__seed-{seed}__model.pt", "model",
                      f"{arch.upper()} behind Figure 1a")
            place(res / f"robot_image_4_{variant}_seed{seed}.data",
                  PAPER / "robot/datasets" / f"{prefix}__seed-{seed}__dataset.data", "dataset", "dataset behind Figure 1a")

    # 3. drop the exploratory duplicate of A
    if A_DUPLICATE.exists():
        for f in A_DUPLICATE.rglob("*"):
            if f.is_file():
                by_path.pop(rel(f), None)
        shutil.rmtree(A_DUPLICATE)

    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(by_path.values(), key=lambda r: r["path"]))
    print("candidate A installed as Figure 1a; B-uniform moved to", rel(B_FOLDER))


if __name__ == "__main__":
    main()
