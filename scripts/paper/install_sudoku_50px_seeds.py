"""Install the 50 px sudoku runs of seeds 174-177 and 180 (all four architectures and the DNN).

Reads the DSMLP outputs copied to results/_incoming/sudoku50_n10/sudoku50_n10_out (run by sudoku50_n10_runner.sh with
the commands, per-cell configs and scoring of the installed cells): CELL2_iv_* for cbm/cem/probcbm, CELL3_iv_* for the
ported ECBM, one file per seed x architecture x tau, and the selective-coverage files with the DNN rows. A cell where a
model cannot reach the selective-accuracy target has no selective columns in the run's output and is not installed.

    python scripts/paper/install_sudoku_50px_seeds.py
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "results/paper"
SRC = REPO / "results/_incoming/sudoku50_n10/sudoku50_n10_out"
SEEDS = (174, 175, 176, 177, 180)
TAUS = {"90": "0.90", "925": "0.925", "95": "0.95", "975": "0.975", "99": "0.99"}
NOTE = "50 px for the seeds that only had 18 px; DSMLP, same commands, configs and scoring as the installed cells"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    index = PAPER / "INDEX.csv"
    with index.open() as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, {r["path"]: r for r in reader}
    installed, skipped = 0, []
    for seed in SEEDS:
        for arch in ("cbm", "cem", "probcbm", "ecbm"):
            for token, tau in TAUS.items():
                src = SRC / f"cells/CELL{'3' if arch == 'ecbm' else '2'}_iv_s{seed}_r50_{arch}_t{token}.csv"
                cells = list(csv.DictReader(src.open())) if src.exists() else []
                if len(cells) != 4 or any(not r.get("coverage_after") for r in cells):
                    skipped.append(f"{arch} seed {seed} tau {tau}")
                    continue
                dest = PAPER / f"sudoku/cells/sudoku__arch-{arch}__res-50px__tau-{tau}__threshold-per-budget__seed-{seed}__interventions.csv"
                shutil.copy2(src, dest)
                rows[str(dest.relative_to(PAPER))] = {"path": str(dest.relative_to(PAPER)), "role": "cited", "sha256": sha256(dest),
                                                     "source": str(src.relative_to(REPO / "results")), "note": NOTE}
                installed += 1
        src = SRC / f"selective/sudoku_selective_tabular_n3_mc9_px50_seed{seed}.csv"
        dest = PAPER / f"sudoku/selective/sudoku__arch-cbm-and-dnn__res-50px__threshold-per-budget__seed-{seed}__selective-all-tau.csv"
        shutil.copy2(src, dest)
        rows[str(dest.relative_to(PAPER))] = {"path": str(dest.relative_to(PAPER)), "role": "cited", "sha256": sha256(dest),
                                             "source": str(src.relative_to(REPO / "results")), "note": NOTE}
    with index.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["path"]))
    print(f"installed {installed} cells and {len(SEEDS)} selective files; not installed (target not reached or missing): {skipped}")


if __name__ == "__main__":
    main()
