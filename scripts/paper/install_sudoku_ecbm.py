"""Install the retrained sudoku ECBM (ported from the authors' code) and archive the standalone ECBM cells.

Reads the DSMLP outputs copied to results/_incoming/sudoku_ecbm: CELL3_iv_s<seed>_r<px>_ecbm_t<tau>.csv (one per
seed x resolution x tau; refit scoring with per-budget decision thresholds) and the trained models. The one cell
where ECBM cannot reach the selective-accuracy target (the pipeline then writes plain accuracy without the
selective columns, "Model ecbm cannot reach target selective accuracy") is installed from the run's own output and
marked in INDEX.csv. Replaced files move to sudoku/archive/ecbm-standalone/ and models/sudoku/archive/ecbm-standalone/.

    python scripts/paper/install_sudoku_ecbm.py
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "results/paper"
SRC = REPO / "results/_incoming/sudoku_ecbm"
CELLS = PAPER / "sudoku/cells"
MODELS = PAPER / "models/sudoku"
TAUS = {"90": "0.90", "925": "0.925", "95": "0.95", "975": "0.975", "99": "0.99"}
RUNS = [(s, 18) for s in range(171, 181)] + [(s, 50) for s in (171, 172, 173, 178, 179)]
UNREACHABLE = {(172, 50, "99")}  # ECBM cannot reach tau=0.99 here; see the run log
NOTE = ("ECBM ported from the authors' code (xmed-lab/ECBM; DSMLP commit 8a6d76f0, serial inference, identical to "
        "the parallel version by scripts/paper/check_ecbm_equivalence.py); refit scoring, per-budget thresholds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    index_path = PAPER / "INDEX.csv"
    with index_path.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = {r["path"]: r for r in reader}

    def rel(p: Path) -> str:
        return str(p.relative_to(PAPER))

    def archive(path: Path, root: Path) -> None:
        dest = root / "archive/ecbm-standalone" / path.name
        if dest.exists():
            raise SystemExit(f"archive already holds {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        entry = rows.pop(rel(path), {"path": rel(path), "sha256": sha256(path), "source": "", "note": ""})
        shutil.move(path, dest)
        rows[rel(dest)] = {**entry, "path": rel(dest), "role": "archive",
                           "note": "standalone ECBM (not the authors' implementation); replaced by the ported ECBM"}

    def install(src: Path, dest: Path, role: str, note: str) -> None:
        if dest.exists():
            archive(dest, PAPER / "sudoku" if dest.parent == CELLS else MODELS)
        shutil.copy2(src, dest)
        rows[rel(dest)] = {"path": rel(dest), "role": role, "sha256": sha256(dest),
                           "source": str(src.relative_to(REPO / "results")), "note": note}

    n = 0
    for seed, px in RUNS:
        for token, tau in TAUS.items():
            dest = CELLS / f"sudoku__arch-ecbm__res-{px}px__tau-{tau}__threshold-per-budget__seed-{seed}__interventions.csv"
            if (seed, px, token) in UNREACHABLE:
                src = SRC / f"cb-sudoku-ecbm/results/sudoku_ecbm_interventions_tabular_n3_mc9_px{px}_seed{seed}.csv"
                log = SRC / f"sudoku_ecbm_out/logs/iv_s{seed}_r{px}_t{token}.txt"
                if "cannot reach target selective accuracy" not in log.read_text():
                    raise SystemExit(f"{log}: expected the 'cannot reach target' message")
                install(src, dest, "cited", NOTE + f"; ECBM cannot reach tau={tau}: plain accuracy, no selective columns")
            else:
                src = SRC / f"sudoku_ecbm_out/cells/CELL3_iv_s{seed}_r{px}_ecbm_t{token}.csv"
                budgets = [r["budget"] for r in csv.DictReader(src.open())]
                if "decision_threshold" not in src.open().readline() or len(budgets) != 4 or budgets[:3] != ["0", "1", "3"]:
                    raise SystemExit(f"{src}: incomplete cell")
                install(src, dest, "cited", NOTE)
            n += 1
        install(SRC / f"cb-sudoku-ecbm/results/sudoku_ecbm_tabular_n3_mc9_px{px}_seed{seed}.model",
                MODELS / f"sudoku__arch-ecbm__res-{px}px__seed-{seed}__model.pt", "model", NOTE)

    logs = PAPER / "sudoku/run_logs/ecbm_retrain"
    for f in sorted(SRC.rglob("*")):
        if f.is_file() and f.suffix in {".txt", ".log", ".sh"}:
            dest = logs / f.relative_to(SRC)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            rows[rel(dest)] = {"path": rel(dest), "role": "log" if f.suffix != ".sh" else "script",
                               "sha256": sha256(dest), "source": str(f.relative_to(REPO / "results")), "note": "sudoku ECBM retrain"}

    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["path"]))
    print(f"installed {n} sudoku ECBM cells and {len(RUNS)} models; INDEX.csv {len(rows)} rows")


if __name__ == "__main__":
    main()
