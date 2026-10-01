"""Install the re-evaluated robot grid into results/paper, archiving the cells it replaces.

The re-evaluation (commit 6f83a214 on `grid-seeded-lfcbm`: k=max uses the same KFlip policy as k<max,
machine-annotated concepts reveal the human ground truth, ProbCBM sampling seeded) wrote one CSV per run
under `<new>/s<seed>/results/`. Every installed cell keeps its file name; its old version moves to
`robot/archive/<archive_name>/` and INDEX.csv is updated for both. Refuses to run unless
scripts/diff_grid_versions.py reports no unexpected differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_diff_module():
    spec = importlib.util.spec_from_file_location("diff_grid_versions", REPO / "scripts/diff_grid_versions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_rows_by_cell(root: Path, diff) -> dict[tuple, tuple[list[str], list[dict], Path]]:
    """(seed, concept_source, intervention_source, family, llm set) -> (header, rows, source file)."""
    cells = {}
    for f in sorted(root.glob("s*/results/*.csv")):
        if f.name.endswith(".bad.csv"):
            raise SystemExit(f"bad output present: {f}")
        seed = f.parent.parent.name.removeprefix("s")
        llm_set = f.stem.split("_")[1] if f.stem.startswith("llm_") else "-"
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (seed, row["concept_source"], row["intervention_source"], row["model_family"], llm_set)
                cells.setdefault(key, (reader.fieldnames, [], f))[1].append(row)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, default=REPO / "results/paper")
    ap.add_argument("--new", type=Path, default=REPO / "results/_incoming/grid_out_v2")
    ap.add_argument("--archive-name", default="grid-sep29-kmax-all-rows-machine-selflabels")
    args = ap.parse_args()

    gate = subprocess.run(
        [sys.executable, str(REPO / "scripts/diff_grid_versions.py"), str(args.paper / "robot/grid"), str(args.new)],
        capture_output=True, text=True,
    )
    print(gate.stdout.strip())
    if gate.returncode != 0:
        raise SystemExit("diff gate failed; nothing installed")

    diff = load_diff_module()
    new_cells = new_rows_by_cell(args.new, diff)
    grid = args.paper / "robot/grid"
    archive = args.paper / "robot/archive" / args.archive_name
    archive.mkdir(parents=True, exist_ok=False)

    index_path = args.paper / "INDEX.csv"
    with index_path.open() as fh:
        reader = csv.DictReader(fh)
        index_fields = reader.fieldnames
        index = list(reader)
    by_path = {r["path"]: r for r in index}

    installed = 0
    for old_file in sorted(grid.glob("*.csv")):
        fields = dict(kv.split("-", 1) for kv in old_file.stem.split("__")[1:-1])
        with old_file.open() as fh:
            first = next(csv.DictReader(fh))
        key = (fields["seed"], first["concept_source"], first["intervention_source"], first["model_family"],
               diff.llm_set_from_old(fields))
        header, rows, source = new_cells[key]

        rel_old = f"robot/grid/{old_file.name}"
        rel_archived = f"robot/archive/{args.archive_name}/{old_file.name}"
        shutil.move(old_file, archive / old_file.name)
        with (grid / old_file.name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: int(r["budget"])))

        old_entry = by_path.pop(rel_old)
        by_path[rel_archived] = {**old_entry, "path": rel_archived, "role": "archive",
                                 "note": f"superseded by re-evaluation (6f83a214); {old_entry['note']}"}
        by_path[rel_old] = {"path": rel_old, "role": "grid", "sha256": sha256(grid / old_file.name),
                            "source": str(source.relative_to(REPO / "results")),
                            "note": f"cell {key[1:4]} re-evaluated at 6f83a214 (one k=max policy, human GT for machine concepts, seeded ProbCBM)"}
        installed += 1

    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=index_fields)
        writer.writeheader()
        writer.writerows(sorted(by_path.values(), key=lambda r: r["path"]))
    print(f"installed {installed} cells; old versions in {archive.relative_to(args.paper)}")


if __name__ == "__main__":
    main()
