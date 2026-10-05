"""Install the diagnostics behind the automation findings into results/paper/.

Three kinds of files, each written by a pipeline option that anyone can rerun:
  1. own-answer cells (robot, balanced rule): interventions that supply the model's own thresholded concept predictions
     (`scripts/robot_pipeline.py --intervention-sources self`), one CSV per concept set x architecture x seed, next to
     the perfect / expert / llm cells in robot/balanced_rule/ (`isrc-self`);
  2. intervention records (robot, balanced rule): per budget, the concepts intervened on, the answers given, the true
     values and the labels before/after (`scripts/robot_pipeline.py --dump-interventions DIR`), in
     robot/balanced_rule/intervention_records/;
  3. confidence files (sudoku): each model's P(valid) on validation and test boards, its concept probabilities, and
     P(valid) after all concepts are set to their true values (`scripts/sudoku_pipeline.py --stages diagnose`), in
     sudoku/confidence/.

Sources are the cluster outputs copied to results/_incoming/ (self_intervention, intervention_dumps, sudoku_probs,
sudoku_confirmed). Existing files are replaced when their content differs.

    python scripts/paper/install_intervention_diagnostics.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "paper"), str(REPO / "scripts")]
from install_balanced_automated import AUTOMATED_SOURCES, BALANCED, BALANCED_TAG, CONCEPTS, PAPER  # noqa: E402

INCOMING = REPO / "results/_incoming"
SEEDS = tuple(range(1014, 1024))
ARCHS = ("cbm", "cem", "probcbm", "ecbm")
RECORDS = BALANCED / "intervention_records"
CONFIDENCE = PAPER / "sudoku/confidence"
SUDOKU_SEEDS = tuple(range(171, 181))
INTERVENERS = {"sim100": "perfect", "sim80": "expert", "llm80": "llm"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arch_tag(arch: str) -> str:
    return "arch-probcbm__mode-joint" if arch == "probcbm" else f"arch-{arch}"


class Index:
    def __init__(self) -> None:
        self.path = PAPER / "INDEX.csv"
        with self.path.open() as fh:
            reader = csv.DictReader(fh)
            self.fields, self.rows = reader.fieldnames, {r["path"]: r for r in reader}

    def add(self, dest: Path, role: str, source: Path | str, note: str) -> None:
        rel = str(dest.relative_to(PAPER))
        source = str(source.relative_to(REPO / "results")) if isinstance(source, Path) else source
        self.rows[rel] = {"path": rel, "role": role, "sha256": sha256(dest), "source": source, "note": note}

    def save(self) -> None:
        with self.path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(sorted(self.rows.values(), key=lambda r: r["path"]))


def install_own_answer_cells(index: Index) -> int:
    """One CSV per concept set x architecture x seed; the runs' `intervention_source` column is rewritten to `self`."""
    src_root = INCOMING / "self_intervention/out"
    n = 0
    for seed in SEEDS:
        for arch in ARCHS:
            files = {"": [f"self_true_{arch}.csv", f"self_human_{arch}.csv"]}
            if arch == "cbm":
                files["koh595"] = ["self_automated_cbm_koh595.csv"]
            cells: dict[tuple[str, str], tuple[Path, list[dict]]] = {}
            for enc, names in files.items():
                for name in names:
                    src = src_root / f"{arch}_s{seed}/results/{name}"
                    for row in csv.DictReader(src.open()):
                        tag = enc or ("hard" if row["concept_source"] in AUTOMATED_SOURCES else "")
                        cells.setdefault((row["concept_source"], tag), (src, []))[1].append({**row, "intervention_source": "self"})
            if set(c for c, _ in cells) != set(CONCEPTS):
                raise SystemExit(f"{arch} seed {seed}: own-answer cells for {sorted(c for c, _ in cells)}, expected all concept sets")
            for (concept_source, enc), (src, rows) in sorted(cells.items()):
                rows.sort(key=lambda r: int(r["budget"]))
                if len(rows) != 4 or [r["budget"] for r in rows][:3] != ["0", "1", "3"]:
                    raise SystemExit(f"{src}: {concept_source} does not have budgets 0, 1, 3, max")
                if any(int(float(r["total_concept_edits_made"])) for r in rows):
                    raise SystemExit(f"{src}: an own-answer intervention changed a concept value")
                enc_tag = f"__enc-{enc}" if enc else ""
                dest = BALANCED / (f"robot__{BALANCED_TAG}__concepts-{CONCEPTS[concept_source]}__{arch_tag(arch)}__isrc-self"
                                   f"__strategy-upto{enc_tag}__seed-{seed}__results.csv")
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                dest.write_text(buffer.getvalue())
                index.add(dest, "cited", src, "interventions with the model's own thresholded concept predictions "
                          "(robot_pipeline.py --intervention-sources self); Bridges")
                n += 1
    return n


def install_intervention_records(index: Index) -> int:
    src_root = INCOMING / "intervention_dumps/out"
    RECORDS.mkdir(parents=True, exist_ok=True)
    n = 0
    for seed in SEEDS:
        for arch in ARCHS:
            for prefix, concepts in (("m7", "true"), ("m12", "human")):
                for token, isrc in INTERVENERS.items():
                    found = sorted((src_root / f"{arch}_s{seed}/dump").glob(f"{prefix}_*__{token}__k*.npz"),
                                   key=lambda f: int(f.stem.split("__k")[1]))
                    if len(found) != 3:
                        raise SystemExit(f"{arch} seed {seed} {concepts} {isrc}: {len(found)} budgets, expected 3")
                    for budget, src in zip(("1", "3", "max"), found):
                        dest = RECORDS / (f"robot__{BALANCED_TAG}__concepts-{concepts}__{arch_tag(arch)}__isrc-{isrc}"
                                          f"__budget-{budget}__seed-{seed}__records.npz")
                        shutil.copy2(src, dest)
                        index.add(dest, "diagnostic", src, "per-robot intervention record "
                                  "(robot_pipeline.py --dump-interventions); accuracies equal the installed cell; Bridges")
                        n += 1
    return n


def install_sudoku_confidence(index: Index) -> int:
    """Merge the two diagnostic runs into the file that `sudoku_pipeline.py --stages diagnose` writes."""
    CONFIDENCE.mkdir(parents=True, exist_ok=True)
    n = 0
    for res in (18, 50):
        for seed in SUDOKU_SEEDS:
            for arch in ARCHS:
                probs = INCOMING / f"sudoku_probs/sudoku_probs_out/probs_{arch}_s{seed}_r{res}.npz"
                confirmed = INCOMING / f"sudoku_confirmed/sudoku_confirmed_out/confirmed_{arch}_s{seed}_r{res}.npz"
                a, b = np.load(probs), np.load(confirmed)
                if not np.array_equal(a["y_test"], b["y"]) or not np.array_equal(a["C_test"], b["C"]):
                    raise SystemExit(f"{arch} seed {seed} {res}px: the two diagnostic runs saw different test boards")
                if not (b["n_checked"] == a["C_test"].shape[1]).all():
                    raise SystemExit(f"{confirmed}: not every concept was set to its true value")
                dest = CONFIDENCE / f"sudoku__arch-{arch}__res-{res}px__seed-{seed}__confidence.npz"
                np.savez_compressed(dest, **{k: a[k] for k in a.files}, p_test_true_concepts=b["p_after"])
                index.add(dest, "diagnostic", probs, "P(valid) on validation/test, concept probabilities, and P(valid) with all "
                          "concepts set to their true values (sudoku_pipeline.py --stages diagnose); DSMLP")
                n += 1
    return n


def main() -> None:
    index = Index()
    counts = (install_own_answer_cells(index), install_intervention_records(index), install_sudoku_confidence(index))
    index.save()
    print(f"installed {counts[0]} own-answer cells, {counts[1]} intervention records, {counts[2]} sudoku confidence files")


if __name__ == "__main__":
    main()
