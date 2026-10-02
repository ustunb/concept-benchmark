"""Install the balanced-rule CBM-vs-DNN results (Figure 1a) and the configuration search into results/paper.

Cited runs: `experiments/skew_sweep.py --dominant-fraction uniform --elbows-weight 3 --is-elbows-concept`
(commit 85637097 on `grid-seeded-lfcbm`), seeds 1014-1023, both concept sets. The July balanced-rule files
they replace move to `robot/archive/july-balanced-undisclosed-patches/` (those runs used patches the paper
does not describe: label noise floor, temperature 4.0, pooled CBM detector). The search runs on pilot seeds
and the two other 10-seed candidates go to `robot/exploratory/balanced-rule-search/`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CITED_TAG = "duniforme3"
CITED_PREFIX = "robot__rule-balanced__sampling-uniform__elbows-concept"
PRESETS = {"ground_truth": ("ideal", "true"), "foot_subtypes": ("subconcept", "human")}
SEARCH = {  # incoming folder -> exploratory subfolder
    "antenna": "antenna-weight-sweep_seeds-2001-2003",
    "skew": "skew-sweep_seeds-2001-2003",
    "skewgrid": "skew-x-elbows-weight_seeds-2004-2006",
    "skewgrid2": "skew-0.30-0.34-x-elbows-weight_seeds-2007-2009",
    "elbows_concept": "elbows-as-concept_seeds-2007-2012",
    "final_d030e2": "candidate-A_elbows-hidden_skew-0.30_seeds-1014-1023",
    "final_B": "candidate-B-0.25_elbows-concept_skew-0.25_seeds-1014-1023",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, default=REPO / "results/paper")
    ap.add_argument("--incoming", type=Path, default=REPO / "results/_incoming")
    args = ap.parse_args()
    paper, incoming = args.paper, args.incoming

    index_path = paper / "INDEX.csv"
    with index_path.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        by_path = {r["path"]: r for r in reader}

    def add(dest: Path, role: str, source: Path, note: str) -> None:
        rel = str(dest.relative_to(paper))
        if rel in by_path:
            raise SystemExit(f"already indexed: {rel}")
        by_path[rel] = {"path": rel, "role": role, "sha256": sha256(dest),
                        "source": str(source.relative_to(REPO / "results")), "note": note}

    def place(source: Path, dest: Path, role: str, note: str) -> None:
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        add(dest, role, source, note)

    # 1. archive the July balanced-rule files
    balanced = paper / "robot/balanced_rule"
    archive = paper / "robot/archive/july-balanced-undisclosed-patches"
    archive.mkdir(parents=True, exist_ok=False)
    for old in sorted(balanced.glob("*.csv")):
        rel_old = str(old.relative_to(paper))
        shutil.move(old, archive / old.name)
        entry = by_path.pop(rel_old)
        rel_new = str((archive / old.name).relative_to(paper))
        by_path[rel_new] = {**entry, "path": rel_new, "role": "archive",
                            "note": "July balanced-rule run with undisclosed patches (noise floor, T=4.0, pooled CBM "
                                    f"detector, uniform sampling, elbows as concept); superseded. {entry['note']}"}

    # 2. cited runs: results, rule checks, logs, models, datasets
    n_cited = 0
    for seed in range(1014, 1024):
        for preset, (variant, name) in PRESETS.items():
            run = incoming / "final_B" / f"run_{CITED_TAG}_s{seed}_{preset}"
            res = run / "results"
            if not (run / "DONE").exists():
                raise SystemExit(f"run not finished: {run}")
            prefix = f"{CITED_PREFIX}__concepts-{name}"
            note = "Figure 1a: balanced rule (appendix weights, elbows weight 3), uniform foot sampling, HasElbows as concept"
            place(next(res.glob(f"robot_{variant}_seed{seed}_*_results.csv")),
                  balanced / f"{prefix}__arch-cbm-and-dnn__seed-{seed}__results.csv", "cited", note)
            place(res / f"robot_image_stochastic_{variant}_cbm_seed{seed}_results.csv",
                  balanced / f"{prefix}__arch-cbm__isrc-perfect__strategy-upto__seed-{seed}__results.csv", "cited", note)
            place(res / "rule_check.json", balanced / f"{prefix}__seed-{seed}__rulecheck.json", "cited",
                  "labeling rule, class balance and training fractions read back from the dataset")
            place(run / "log.txt", balanced / "run_logs" / f"{prefix}__seed-{seed}__log.txt", "log", "run log")
            for arch in ("cbm", "dnn"):
                place(res / f"robot_image_stochastic_4_{variant}_{arch}_seed{seed}.model",
                      paper / "models/robot/balanced" / f"{prefix}__arch-{arch}__seed-{seed}__model.pt", "model",
                      f"{arch.upper()} behind Figure 1a")
            place(res / f"robot_image_4_{variant}_seed{seed}.data",
                  paper / "robot/datasets" / f"{prefix}__seed-{seed}__dataset.data", "dataset", "dataset behind Figure 1a")
            n_cited += 1

    # 3. configuration search and the other 10-seed candidates (results only)
    search_root = paper / "robot/exploratory/balanced-rule-search"
    n_search = 0
    for folder, sub in SEARCH.items():
        for run in sorted((incoming / folder).glob("run_*")):
            if folder == "final_B" and not run.name.startswith("run_d0.25e3_"):
                continue
            for f in [*sorted((run / "results").glob("*.csv")), run / "results/rule_check.json", run / "log.txt"]:
                if f.exists():
                    place(f, search_root / sub / run.name / f.name, "exploratory", f"balanced-rule configuration search ({sub})")
            n_search += 1

    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(by_path.values(), key=lambda r: r["path"]))
    print(f"cited runs installed: {n_cited}; search runs stored: {n_search}; July files archived in {archive.relative_to(paper)}")


if __name__ == "__main__":
    main()
