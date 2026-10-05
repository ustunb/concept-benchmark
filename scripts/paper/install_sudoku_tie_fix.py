"""Install the sudoku cells re-scored with the tie fix and archive the cells they replace.

The abstention threshold is fitted on validation data keeping predictions exactly at the threshold, but the rule applied
at test time (and by the intervention strategies) deferred them. Models that give many boards the same probability
(ECBM, CEM) lost those boards. scripts/sudoku_pipeline.py now returns a threshold that keeps them
(`_selective_accuracy_threshold`); every cell was re-scored on DSMLP with the same models, data, configs and commands
(results/_incoming/sudoku_tiefix: CELL4_iv_s<seed>_r<px>_<arch>_t<tau>.csv). Replaced files move to
sudoku/archive/before-tie-fix/. A cell where a model cannot reach the target has no selective columns and is removed.

    python scripts/paper/install_sudoku_tie_fix.py
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "results/paper"
SRC = REPO / "results/_incoming/sudoku_tiefix/sudoku_tiefix_out/cells"
CELLS = PAPER / "sudoku/cells"
ARCHIVE = PAPER / "sudoku/archive/before-tie-fix"
TAUS = {"90": "0.90", "925": "0.925", "95": "0.95", "975": "0.975", "99": "0.99"}
NOTE = "re-scored with the tie fix (threshold fitted and applied with the same rule); DSMLP, same models, data and configs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    index = PAPER / "INDEX.csv"
    with index.open() as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, {r["path"]: r for r in reader}
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    installed = changed = 0
    seen = set()
    for src in sorted(SRC.glob("CELL4_iv_s*_r*_*_t*.csv")):
        _, _, s, r, arch, t = src.stem.split("_")
        dest = CELLS / f"sudoku__arch-{arch}__res-{r[1:]}px__tau-{TAUS[t[1:]]}__threshold-per-budget__seed-{s[1:]}__interventions.csv"
        seen.add(dest.name)
        rel = str(dest.relative_to(PAPER))
        if dest.exists():
            if sha256(dest) == sha256(src):
                installed += 1
                continue
            old = ARCHIVE / dest.name
            shutil.move(dest, old)
            entry = rows.pop(rel, {"path": rel, "sha256": sha256(old), "source": "", "note": ""})
            rows[str(old.relative_to(PAPER))] = {**entry, "path": str(old.relative_to(PAPER)), "role": "archive",
                                                 "note": (entry.get("note", "") + "; replaced by the tie-fix re-score").lstrip("; ")}
            changed += 1
        shutil.copy2(src, dest)
        rows[rel] = {"path": rel, "role": "cited", "sha256": sha256(dest), "source": str(src.relative_to(REPO / "results")), "note": NOTE}
        installed += 1
    removed = []
    for dest in sorted(CELLS.glob("sudoku__arch-*__interventions.csv")):
        if dest.name not in seen:  # the re-score could not reach the target for this cell
            old = ARCHIVE / dest.name
            rel = str(dest.relative_to(PAPER))
            shutil.move(dest, old)
            entry = rows.pop(rel, {"path": rel, "sha256": sha256(old), "source": "", "note": ""})
            rows[str(old.relative_to(PAPER))] = {**entry, "path": str(old.relative_to(PAPER)), "role": "archive",
                                                 "note": (entry.get("note", "") + "; target not reached in the tie-fix re-score").lstrip("; ")}
            removed.append(dest.name)
    with index.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["path"]))
    print(f"installed {installed} cells ({changed} replaced a different file, archived in {ARCHIVE.relative_to(PAPER)}); removed {removed}")


if __name__ == "__main__":
    main()
