"""Install the section 3 rerun on the balanced rule and the Table 3 seed-1014 LLM-intervention cells.

Reads the per-unit downloads in results/_incoming/sec3_by_seed/ (one folder per architecture x seed,
<arch>_<seed>_<cluster>/sec3/out/<arch>_s<seed>/results/*.csv, plus t3_<cluster>/grid1014_llm/out/results/*.csv).
Each run CSV is checked against the cells it must hold (exact concept_source x intervention_source pairs, budgets
0, 1, 3, max) and split into one file per cell:
  - section 3 -> robot/balanced_rule/ (rule-balanced, skew 0.30, elbows weight 2), next to section 1's CBM files;
    an existing file must have identical content (section 1's CBM true/human perfect cells are reproduced exactly);
  - Table 3 seed 1014 -> robot/grid/ (rule-sparse) under the names seeds 1015-1017 use.
Models trained in the rerun (results/_incoming/sec3_models_bridges) go to models/robot/balanced/, run logs to
robot/balanced_rule/run_logs/sec3/. INDEX.csv records the source of every file.

    python scripts/paper/install_balanced_automated.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "results/paper"
BALANCED = PAPER / "robot/balanced_rule"
GRID = PAPER / "robot/grid"
MODELS = PAPER / "models/robot/balanced"
INCOMING = REPO / "results/_incoming/sec3_by_seed"
MODELS_IN = REPO / "results/_incoming/sec3_models_bridges"
SEEDS = tuple(range(1014, 1024))
# runs replaced after the fact; their installed cells come from another installer
SUPERSEDED = {("probcbm", 1014): ("true_probcbm", "llm_true_probcbm")}  # install_probcbm_1014_retrain.py
SUPERSEDED_MODELS = {"ideal_probcbm_seed1014"}
ARCHS = ("cbm", "cem", "probcbm", "ecbm")
CONCEPTS = {"ground_truth": "true", "human_concepts": "human", "machine_annotation": "machine",
            "llm_concepts": "llm", "clip_concepts": "clip"}
AUTOMATED_SOURCES = ("machine_annotation", "llm_concepts", "clip_concepts")
LLM_TAG = "isrc-llm__llm-gemini-2.5-flash-lite__img-224px__questions-v2-value-explicit"
BALANCED_TAG = "rule-balanced__sampling-skew0.30__elbows-weight2"


def section3_runs(arch: str) -> dict[str, tuple[set, str]]:
    """run name -> (expected cells, encoding tag for automated concept sets)."""
    pe = ("perfect", "expert")
    runs = {
        f"true_{arch}": ({("ground_truth", i) for i in pe}, ""),
        f"human_{arch}": ({("human_concepts", i) for i in pe}, ""),
        f"automated_{arch}": ({(c, i) for c in AUTOMATED_SOURCES for i in pe}, "hard"),
        f"llm_true_{arch}": ({("ground_truth", "llm")}, ""),
        f"llm_human_{arch}": ({(c, "llm") for c in ("human_concepts", *AUTOMATED_SOURCES)}, "hard"),
    }
    if arch == "cbm":
        runs["automated_cbm_koh595"] = ({(c, i) for c in AUTOMATED_SOURCES for i in pe}, "koh595")
        runs["llm_automated_cbm_koh595"] = ({(c, "llm") for c in AUTOMATED_SOURCES}, "koh595")
    return runs


def table3_runs() -> dict[str, set]:
    return {**{f"llm_ideal_{a}": {("ground_truth", "llm")} for a in ARCHS},
            **{f"llm_subconcept_{a}": {(c, "llm") for c in ("human_concepts", *AUTOMATED_SOURCES)} for a in ARCHS}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checked(path: Path, expected: set, arch: str) -> dict:
    rows = list(csv.DictReader(path.open()))
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["concept_source"], r["intervention_source"]), []).append(r)
    budgets_ok = all([r["budget"] for r in v][:3] == ["0", "1", "3"] and len(v) == 4 for v in cells.values())
    if set(cells) != expected or not budgets_ok or {r["model_family"] for r in rows} != {arch}:
        raise SystemExit(f"{path}: incomplete or unexpected cells {sorted(cells)}")
    return cells


def cell_name(rule: str, concept_source: str, isrc: str, arch: str, enc: str, seed: int) -> str:
    concepts = CONCEPTS[concept_source]
    arch_tag = "arch-probcbm__mode-joint" if arch == "probcbm" else f"arch-{arch}"
    src = LLM_TAG if isrc == "llm" else f"isrc-{isrc}"
    enc_tag = f"__enc-{enc}" if enc and concept_source in AUTOMATED_SOURCES else ""
    return f"robot__{rule}__concepts-{concepts}__{arch_tag}__{src}__strategy-upto{enc_tag}__seed-{seed}__results.csv"


def unit_dir(name: str) -> Path:
    found = [d for d in INCOMING.glob(f"{name}_*") if d.is_dir() and not d.name.endswith(".partial")]
    if len(found) != 1:
        raise SystemExit(f"expected one downloaded unit for {name}, found {[d.name for d in found]}")
    return found[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate the downloads only, write nothing")
    args = ap.parse_args()

    index_path = PAPER / "INDEX.csv"
    with index_path.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = {r["path"]: r for r in reader}

    def rel(p: Path) -> str:
        return str(p.relative_to(PAPER))

    def install_cells(src: Path, cells: dict, dest_dir: Path, name_of, note: str) -> int:
        n = 0
        for (concept_source, isrc), cell_rows in sorted(cells.items()):
            dest = dest_dir / name_of(concept_source, isrc)
            tmp = dest.with_name(dest.name + ".tmp")
            with tmp.open("w", newline="") as fh:
                # match the folder: balanced_rule files end lines with \n, the grid installers wrote \r\n
                w = csv.DictWriter(fh, fieldnames=list(cell_rows[0]), lineterminator="\n" if dest_dir == BALANCED else "\r\n")
                w.writeheader()
                w.writerows(cell_rows)
            if dest.exists():
                if sha256(dest) != sha256(tmp):
                    tmp.unlink()
                    raise SystemExit(f"{dest.name} exists with different content; not overwritten")
                tmp.unlink()
                old = rows.get(rel(dest), {})
                mark = "reproduced exactly by the section 3 rerun"
                if old.get("source") != str(src.relative_to(REPO / "results")) and mark not in old.get("note", ""):
                    rows[rel(dest)] = {**old, "path": rel(dest), "note": (old.get("note", "") + "; " + mark).lstrip("; ")}
            else:
                tmp.rename(dest)
                rows[rel(dest)] = {"path": rel(dest), "role": "cited", "sha256": sha256(dest),
                                   "source": str(src.relative_to(REPO / "results")), "note": note}
            n += 1
        return n

    plan = []  # (src, cells, dest_dir, name_of, note)
    for seed in SEEDS:
        for arch in ARCHS:
            d = unit_dir(f"{arch}_{seed}")
            cluster = d.name.rsplit("_", 1)[1]
            note = (f"section 3 rerun on the balanced rule (Bridges prep_s{seed} label-free CBMs); run on {cluster}"
                    + ("; true/human models from section 1" if arch == "cbm" else ""))
            for run, (expected, enc) in section3_runs(arch).items():
                if run in SUPERSEDED.get((arch, seed), ()):
                    continue
                src = d / f"sec3/out/{arch}_s{seed}/results/{run}.csv"
                cells = read_checked(src, expected, arch)
                run_note = note + ("; label-free CBM interventions in-distribution (5th/95th percentile)"
                                   if enc == "koh595" else "")
                plan.append((src, cells, BALANCED,
                             lambda c, i, a=arch, e=enc, s=seed: cell_name(BALANCED_TAG, c, i, a, e, s), run_note))
    t3 = unit_dir("t3")
    t3_cluster = t3.name.rsplit("_", 1)[1]
    for run, expected in table3_runs().items():
        arch = run.rsplit("_", 1)[1]
        src = t3 / f"grid1014_llm/out/results/{run}.csv"
        cells = read_checked(src, expected, arch)
        plan.append((src, cells, GRID, lambda c, i, a=arch: cell_name("rule-sparse", c, i, a, "hard", 1014),
                     f"Table 3 seed 1014 LLM interventions (Gemini 2.5 Flash-Lite, 224 px, cache only) on the grid "
                     f"models; run on {t3_cluster}"))
    n_cells = sum(len(p[1]) for p in plan)
    print(f"checked {len(plan)} run files, {n_cells} cells")
    if args.check:
        return

    installed = sum(install_cells(*p) for p in plan)

    MODELS.mkdir(parents=True, exist_ok=True)
    n_models = 0
    for f in sorted(MODELS_IN.rglob("*.model")):
        stem = f.name.removeprefix("robot_image_stochastic_4_").removesuffix(".model")
        if stem in SUPERSEDED_MODELS:
            continue
        dest = MODELS / f"robot__{BALANCED_TAG}__{stem}__trained-2026-10__model.pt"
        shutil.copy2(f, dest)
        rows[rel(dest)] = {"path": rel(dest), "role": "model", "sha256": sha256(dest),
                           "source": str(f.relative_to(REPO / "results")), "note": "section 3 rerun (Bridges)"}
        n_models += 1

    logs = BALANCED / "run_logs/sec3"
    for f in sorted(INCOMING.rglob("*")):
        if f.is_file() and f.suffix in {".txt", ".log"}:
            dest = logs / f.relative_to(INCOMING)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            rows[rel(dest)] = {"path": rel(dest), "role": "log", "sha256": sha256(dest),
                               "source": str(f.relative_to(REPO / "results")), "note": "section 3 rerun"}

    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["path"]))
    print(f"installed {installed} cells, {n_models} models; INDEX.csv {len(rows)} rows")


if __name__ == "__main__":
    main()
