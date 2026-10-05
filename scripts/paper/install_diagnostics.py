"""Install the mechanism diagnostics behind sections 1 and 2 into results/paper and index them.

Source: results/_incoming/bridges_diag (copied from Bridges) and results/_incoming/diagnostics. Destinations:
  robot/diagnostics/                      subtype check, intervention response, ProbCBM and ECBM diagnostics
  robot/diagnostics/probcbm-retrain/      ProbCBM causal test: control, fixed prototypes, independent, vib_beta=0
  robot/diagnostics/run_scripts/          Slurm jobs and worker scripts used on Bridges
  models/robot/diagnostics/               the retrained ProbCBM models of the causal test
Also indexes files already in results/paper that INDEX.csv does not list (grid run logs and scripts,
grid_table.txt, figures). Refuses to overwrite an existing file.

    python scripts/paper/install_diagnostics.py
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "results/paper"
SRC = REPO / "results/_incoming/bridges_diag"
DIAG = PAPER / "robot/diagnostics"
SEEDS = (1014, 1015, 1016, 1017)
VARIANTS = {
    "control": "unmodified ProbCBM retrained (grid code 6f83a214, Bridges V100)",
    "fixed": "ProbCBM with absent = -present prototypes, frozen (scripts/paper/train_probcbm_fixed_prototypes.py)",
    "independent": "ProbCBM with training_mode=independent (label loss cut off from the detector)",
    "novib": "ProbCBM with vib_beta=0",
}


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

    def place(source: Path, dest: Path, role: str, note: str) -> None:
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        rows[rel(dest)] = {"path": rel(dest), "role": role, "sha256": sha256(dest),
                           "source": str(source.relative_to(REPO / "results")), "note": note}

    balanced = "robot__rule-balanced__sampling-skew0.30__elbows-weight2"
    place(SRC / "diag/subtypes/slurm_47314799.out",
          DIAG / f"{balanced}__concepts-human__arch-cbm__unlisted-subtype-check__log.txt", "cited",
          "section 1: full intervention on listed vs unlisted foot subtypes, detector firing on unlisted subtypes, "
          "k=0 accuracy by group (scripts/paper/check_unlisted_subtypes.py --images)")
    for seed in SEEDS:
        place(SRC / f"diag/intervention_response_{seed}.csv",
              DIAG / f"robot__rule-sparse__arch-all__intervention-response__seed-{seed}.csv", "cited",
              "section 2: detection, single-concept and all-concept interventions, majority share "
              "(scripts/paper/measure_intervention_response.py)")
    place(SRC / "diag/probcbm/probcbm_response.csv",
          DIAG / "robot__rule-sparse__arch-probcbm-and-cbm__intervention-curves.csv", "cited",
          "section 2: accuracy vs number of random concepts intervened, sampled vs mean embeddings, "
          "listed vs unlisted subtypes (scripts/paper/measure_probcbm_response.py)")
    place(SRC / "diag/probcbm/probcbm_heads.csv",
          DIAG / "robot__rule-sparse__concepts-true__arch-probcbm__prototypes-and-head.csv", "cited",
          "section 2: present/absent prototype cosine and head sensitivity with all concepts true "
          "(scripts/paper/measure_probcbm_heads.py)")
    place(SRC / "diag/probcbm/detection_grid.csv",
          DIAG / "robot__rule-sparse__concepts-true__arch-probcbm__detection-train-test.csv", "cited",
          "section 2: ProbCBM concept detection on training vs test robots (scripts/paper/measure_probcbm_detection.py)")
    place(SRC / "ecbm_diag/ecbm_response.csv",
          DIAG / "robot__rule-sparse__concepts-true__arch-ecbm__response.csv", "cited",
          "section 2: ported ECBM, image-only vs concept-path accuracy, switch-only test, random-subset curves "
          "(scripts/paper/measure_ecbm_response.py)")

    for variant, note in VARIANTS.items():
        base = SRC / f"pcbm_test/{variant}"
        for seed in SEEDS:
            stem = f"robot__rule-sparse__concepts-true__arch-probcbm__variant-{variant}__seed-{seed}"
            run = base / f"run_s{seed}/results"
            place(run / f"robot_image_stochastic_ideal_probcbm_seed{seed}_labels12345_results.csv",
                  DIAG / f"probcbm-retrain/{stem}__results.csv", "cited", note)
            place(base / f"heads_s{seed}.csv", DIAG / f"probcbm-retrain/{stem}__prototypes-and-head.csv", "cited", note)
            place(base / f"detection_s{seed}.csv", DIAG / f"probcbm-retrain/{stem}__detection-train-test.csv", "cited", note)
            place(run / f"robot_image_stochastic_4_ideal_probcbm_seed{seed}_labels12345.model",
                  PAPER / f"models/robot/diagnostics/{stem}__model.pt", "model", note)

    for script in sorted(SRC.rglob("*")):
        if script.is_file() and script.suffix in {".sh", ".slurm", ".py"}:
            place(script, DIAG / "run_scripts" / script.relative_to(SRC).as_posix().replace("/", "__"),
                  "script", "Bridges job or worker script for the diagnostics")

    for p in sorted(PAPER.rglob("*")):  # files already in place but not indexed
        if not p.is_file() or p.name in ("INDEX.csv", "README.md", ".DS_Store") or rel(p).startswith("images/"):
            continue
        if rel(p) not in rows:
            role = "log" if "/run_logs/" in rel(p) else "script" if "/run_scripts/" in rel(p) else \
                "figure" if rel(p).startswith("figures/") else "derived"
            rows[rel(p)] = {"path": rel(p), "role": role, "sha256": sha256(p), "source": "", "note": "indexed on install"}

    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["path"]))
    print(f"INDEX.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
