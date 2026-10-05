"""Install the retrained ECBM (authors' code, ported) into the robot grid, archiving the standalone ECBM cells.

Reads the Bridges run outputs copied to results/_incoming/ecbm_retrain_dl (ecbm_out/s<seed>/results/*.csv, one CSV
per run holding several cells; run_ecbm_s<seed>/results/*ecbm*.model). Each run CSV is checked against the cells
it must hold (exact concept_source x intervention_source pairs, budgets 0, 1, 3, max) and split into the grid's
one-file-per-cell layout under the existing names. The replaced files move to robot/archive/<archive>/ and
models/robot/archive/<archive>/; the standalone ECBM's 32 px and April LLM-intervention cells, which the
retrain does not reproduce, move there too. INDEX.csv is updated for every file. Runs that are not finished
can be left out with --skip seed:run and installed later with a second call.

    python scripts/paper/install_ecbm_retrain.py --skip 1017:llm_res224_subconcept
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "results/paper"
GRID = PAPER / "robot/grid"
MODELS = PAPER / "models/robot/grid"
INCOMING = REPO / "results/_incoming/ecbm_retrain_dl"
SEEDS = (1014, 1015, 1016, 1017)
CONCEPTS = {"ground_truth": "true", "human_concepts": "human", "machine_annotation": "machine",
            "llm_concepts": "llm", "clip_concepts": "clip"}
AUTOMATED = {"machine", "llm", "clip"}
LLM_TAG = "isrc-llm__llm-gemini-2.5-flash-lite__img-224px__questions-v2-value-explicit"
RUNS = {  # run file -> expected (concept_source, intervention_source) cells
    "ecbm_ideal": {("ground_truth", "perfect"), ("ground_truth", "expert")},
    "ecbm_subconcept": {("human_concepts", "perfect"), ("human_concepts", "expert")},
    "ecbm_subconcept_automated": {(c, i) for c in ("machine_annotation", "llm_concepts", "clip_concepts")
                                  for i in ("perfect", "expert")},
    "llm_res224_ideal": {("ground_truth", "llm")},
    "llm_res224_subconcept": {(c, "llm") for c in ("human_concepts", "machine_annotation", "llm_concepts",
                                                    "clip_concepts")},
}
MODEL_FILES = {"ideal_ecbm": "true", "subconcept_ecbm": "human", "subconcept_ecbm_machine": "machine",
               "subconcept_ecbm_llm": "llm", "subconcept_ecbm_clip": "clip"}
NOTE = ("ECBM ported from the authors' code (xmed-lab/ECBM; branch commit 8ea4489f): authors' energy network, losses, "
        "concept augmentation, gradient inference and intervention procedure; shared CNN backbone, hidden 64, Adam")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_name(concepts: str, isrc: str, seed: int) -> str:
    enc = "__enc-hard" if concepts in AUTOMATED else ""
    src = LLM_TAG if isrc == "llm" else f"isrc-{isrc}"
    return f"robot__rule-sparse__concepts-{concepts}__arch-ecbm__{src}__strategy-upto{enc}__seed-{seed}__results.csv"


def read_checked(path: Path, expected: set) -> dict:
    rows = list(csv.DictReader(path.open()))
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["concept_source"], r["intervention_source"]), []).append(r)
    budgets_ok = all([r["budget"] for r in v][:3] == ["0", "1", "3"] and len(v) == 4 for v in cells.values())
    if set(cells) != expected or not budgets_ok or {r["model_family"] for r in rows} != {"ecbm"}:
        raise SystemExit(f"{path}: incomplete or unexpected cells {sorted(cells)}")
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="seed:run pairs not finished yet (e.g. 1017:llm_res224_subconcept)")
    ap.add_argument("--archive-name", default="ecbm-standalone-2026-07")
    args = ap.parse_args()
    skip = {tuple(s.split(":")) for s in args.skip}

    index_path = PAPER / "INDEX.csv"
    with index_path.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = {r["path"]: r for r in reader}

    def rel(p: Path) -> str:
        return str(p.relative_to(PAPER))

    def archive(path: Path, archive_root: Path) -> None:
        dest = archive_root / args.archive_name / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        entry = rows.pop(rel(path), {"path": rel(path), "sha256": sha256(path), "source": "", "note": ""})
        shutil.move(path, dest)
        rows[rel(dest)] = {**entry, "path": rel(dest), "role": "archive",
                           "note": "standalone ECBM (not the authors' implementation); replaced by the ported ECBM"}

    def install(dest: Path, write, role: str, source: str) -> None:
        tmp = dest.with_name(dest.name + ".tmp")
        write(tmp)
        if dest.exists():
            if sha256(dest) == sha256(tmp):  # already installed, or the old file has identical content
                tmp.unlink()
                same = rows.get(rel(dest), {}).get("note", "") != NOTE
                rows[rel(dest)] = {"path": rel(dest), "role": role, "sha256": sha256(dest), "source": source,
                                   "note": NOTE + ("; identical to the standalone ECBM's output" if same else "")}
                return
            archive(dest, dest.parent.parent / "archive" if dest.parent == GRID else PAPER / "models/robot/archive")
        tmp.rename(dest)
        rows[rel(dest)] = {"path": rel(dest), "role": role, "sha256": sha256(dest), "source": source, "note": NOTE}

    installed = 0
    for seed in SEEDS:
        for run, expected in RUNS.items():
            if run.startswith("llm") and seed == 1014 or (str(seed), run) in skip:
                continue
            src = INCOMING / f"ecbm_out/s{seed}/results/{run}.csv"
            cells = read_checked(src, expected)
            for (concept_source, isrc), cell_rows in sorted(cells.items()):
                header = list(cell_rows[0])

                def write(dest, cell_rows=cell_rows, header=header):
                    with dest.open("w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=header)
                        w.writeheader()
                        w.writerows(cell_rows)

                install(GRID / cell_name(CONCEPTS[concept_source], isrc, seed), write, "cited",
                        str(src.relative_to(REPO / "results")))
                installed += 1
        for stem, concepts in MODEL_FILES.items():
            src = INCOMING / f"run_ecbm_s{seed}/results/robot_image_stochastic_4_{stem}_seed{seed}_labels12345.model"
            old = list(MODELS.glob(f"robot__rule-sparse__concepts-{concepts}__arch-ecbm__trained-*__seed-{seed}__model.pt"))
            for o in old:
                if "trained-2026-10" not in o.name:
                    archive(o, PAPER / "models/robot/archive")
            install(MODELS / f"robot__rule-sparse__concepts-{concepts}__arch-ecbm__trained-2026-10__seed-{seed}__model.pt",
                    lambda d, s=src: shutil.copy2(s, d), "model", str(src.relative_to(REPO / "results")))

    for old in sorted(GRID.glob("*__arch-ecbm__isrc-llm__*")):  # 32 px and April LLM cells are not reproduced
        if "img-224px" not in old.name:
            archive(old, GRID.parent / "archive")

    logs = PAPER / "robot/grid/run_logs/ecbm_retrain"
    for f in sorted(INCOMING.rglob("*")):
        if f.is_file() and (f.suffix in {".txt", ".log", ".sh", ".slurm"}) and "results" not in f.parts:
            dest = logs / f.relative_to(INCOMING)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            rows[rel(dest)] = {"path": rel(dest), "role": "log" if f.suffix in {".txt", ".log"} else "script",
                               "sha256": sha256(dest), "source": str(f.relative_to(REPO / "results")), "note": "ECBM retrain"}

    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["path"]))
    print(f"installed {installed} ECBM cells; INDEX.csv {len(rows)} rows")


if __name__ == "__main__":
    main()
