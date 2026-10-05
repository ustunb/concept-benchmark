"""Replace the balanced-rule ProbCBM, seed 1014, true concepts, with the DSMLP retrain.

The Bridges section 3 run of this model collapsed to one class (50.1% at k = 0). Retraining with the same code, data and
seed gives 80.2 / 82.7 / 83.8 / 89.7 on DSMLP, identically on three runs: ProbCBM's training outcome depends on the GPU's
arithmetic (V100 vs A30), not on the seed. Each cell in the given run CSV (true concepts x intervention source) replaces
the installed balanced-rule file; the replaced file moves to robot/archive/<archive>/ and INDEX.csv records both.

    python scripts/paper/install_probcbm_1014_retrain.py --run-dir results/_incoming/probcbm_1014_dsmlp
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from install_balanced_automated import BALANCED, BALANCED_TAG, MODELS, PAPER, REPO, cell_name, sha256

ARCHIVE = PAPER / "robot/archive"
NOTE = ("ProbCBM seed 1014 retrained on DSMLP (A30), same code/data/seed as the Bridges run, which collapsed to one class; "
        "three DSMLP retrains give identical accuracies")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--archive-name", default="probcbm-seed1014-bridges-v100")
    args = ap.parse_args()
    args.run_dir = args.run_dir.resolve()

    index_path = PAPER / "INDEX.csv"
    with index_path.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = {r["path"]: r for r in reader}

    def rel(p: Path) -> str:
        return str(p.relative_to(PAPER))

    def archive(path: Path) -> None:
        dest = ARCHIVE / args.archive_name / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        entry = rows.pop(rel(path), {"path": rel(path), "sha256": sha256(path), "source": "", "note": ""})
        shutil.move(path, dest)
        rows[rel(dest)] = {**entry, "path": rel(dest), "role": "archive",
                           "note": "Bridges (V100) ProbCBM seed 1014, collapsed to one class; replaced by the DSMLP retrain"}

    cells: dict = {}
    for src in sorted((args.run_dir / "results").glob("*_ideal_probcbm_seed1014_*results.csv")):
        for r in csv.DictReader(src.open()):
            cells.setdefault(r["intervention_source"], (src, []))[1].append(r)
    for isrc, (src, cell_rows) in sorted(cells.items()):
        if [r["budget"] for r in cell_rows][:3] != ["0", "1", "3"] or len(cell_rows) != 4 or \
                {r["concept_source"] for r in cell_rows} != {"ground_truth"} or {r["model_family"] for r in cell_rows} != {"probcbm"}:
            raise SystemExit(f"{src}: incomplete cell {isrc}")
        dest = BALANCED / cell_name(BALANCED_TAG, "ground_truth", isrc, "probcbm", "", 1014)
        if dest.exists():
            archive(dest)
        with dest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cell_rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(cell_rows)
        rows[rel(dest)] = {"path": rel(dest), "role": "cited", "sha256": sha256(dest),
                           "source": str(src.relative_to(REPO / "results")), "note": NOTE}
        print(f"installed {dest.name}: {[round(100 * float(r['accuracy']), 1) for r in cell_rows]}")

    model_src = args.run_dir / "results/robot_image_stochastic_4_ideal_probcbm_seed1014.model"
    model_dest = MODELS / f"robot__{BALANCED_TAG}__ideal_probcbm_seed1014__trained-2026-10__model.pt"
    if model_dest.exists() and sha256(model_dest) != sha256(model_src):
        archive_dest = PAPER / "models/robot/archive" / args.archive_name / model_dest.name
        archive_dest.parent.mkdir(parents=True, exist_ok=True)
        entry = rows.pop(rel(model_dest), {"path": rel(model_dest), "sha256": sha256(model_dest), "source": "", "note": ""})
        shutil.move(model_dest, archive_dest)
        rows[rel(archive_dest)] = {**entry, "path": rel(archive_dest), "role": "archive", "note": "Bridges (V100) ProbCBM seed 1014"}
    shutil.copy2(model_src, model_dest)
    rows[rel(model_dest)] = {"path": rel(model_dest), "role": "model", "sha256": sha256(model_dest),
                             "source": str(model_src.relative_to(REPO / "results")), "note": NOTE}

    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["path"]))
    print(f"INDEX.csv {len(rows)} rows")


if __name__ == "__main__":
    main()
