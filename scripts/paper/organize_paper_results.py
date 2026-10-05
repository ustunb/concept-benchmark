"""Build one self-named tree of every paper result, model, cache, dataset and image.

Sources (read-only here): `results/paper/_raw/` (DSMLP backups mirrored Sep 29), `results/camera_ready/`,
and `results/_incoming/` (the Sep 29–30 grid, its models and datasets, LLM caches, images, July DSMLP
models). Output: `results/paper_tree/` plus `INDEX.csv`. Nothing is deleted here; `--verify` checks that
every source file is represented in the tree (placed, or a byte-identical duplicate of a placed file).

Naming: `<benchmark>__<field>-<value>__…__<kind>.<ext>`. Fields come from file contents where the file
carries them (concept_source, intervention_source, model_family, budgets); otherwise from run scripts,
logs and timestamps recorded below — never guessed. Two different files mapping to one name is a hard
error. Original paths live only in INDEX.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
RAW = RESULTS / "paper" / "_raw"
CAMERA = RESULTS / "camera_ready"
INCOMING = RESULTS / "_incoming"
OLD_PAPER = RESULTS / "paper"
OUT = RESULTS / "paper_tree"

CONCEPTS = {
    "ground_truth": "true",
    "human_concepts": "human",
    "machine_annotation": "machine",
    "llm_concepts": "llm",
    "clip_concepts": "clip",
}
PRESET = {"ideal": "true", "subconcept": "human", "ground_truth": "true", "foot_subtypes": "human"}
ENC = {"hard": "hard", "hard_revealed": "hardrev", "koh595": "koh595"}
TAU = {"90": "0.90", "925": "0.925", "95": "0.95", "975": "0.975", "99": "0.99"}
ARCH = {"cs": "cbm", "cbm": "cbm", "cem": "cem", "probcbm": "probcbm", "ecbm": "ecbm", "dnn": "dnn", "lfcbm": "lfcbm"}

# Figure 1a: balanced-rule collect files the paper uses (camera_ready/aggregate.py BAL).
BALANCED_PINNED = {
    "ideal": dict(zip(map(str, range(1014, 1024)), "d6739862 5a9b4d61 5eefd147 6c3054d1 985014e4 3bb6e761 019507ca 861e8c5e 03747bef 1777a829".split())),
    "subconcept": dict(zip(map(str, range(1014, 1024)), "7b0ed1be 38c57466 92ca4405 f91b796d 420dac6f 5e9def59 da75cdca 6e84cc43 dcf72706 e3ea2eed".split())),
}

# LLM-vote caches: concept-name sha1 prefix -> concept set it answers
CACHE_CONCEPTS = {"e2d4e470": "true", "ab9271e4": "human-and-machine", "1bc3dda2": "llm", "dab318a1": "clip"}
# test-set sha1 prefix -> seed (verified against the July datasets, Sep 29)
TEST_SET_SEED = {"7f958bf6": "1014", "51687eb3": "1015", "a92eabf5": "1016", "91a650f3": "1017"}

ORDER = ["rule", "concepts", "arch", "by", "mode", "isrc", "llm", "img", "questions", "strategy", "enc",
         "res", "tau", "threshold", "trained", "seed", "cfg", "saved"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def name(benchmark: str, fields: dict, kind: str, ext: str) -> str:
    assert all("-" not in k for k in fields), "field names must not contain hyphens (names parse on the first one)"
    parts = [f"{k}-{fields[k]}" for k in ORDER if fields.get(k) not in (None, "")]
    unknown = set(fields) - set(ORDER)
    assert not unknown, f"unknown fields {unknown}"
    return "__".join([benchmark, *parts, kind]) + ext


def saved(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d")


def read_cells(p: Path) -> dict[tuple, list[dict]]:
    rows = list(csv.DictReader(p.open()))
    out = defaultdict(list)
    for r in rows:
        out[(r.get("concept_source"), r.get("intervention_source"), r.get("model_family"))].append(r)
    return out


# ───────────────────────── placement records ─────────────────────────

class Tree:
    def __init__(self, dry: bool):
        self.dry = dry
        self.dest_hash: dict[Path, str] = {}   # dest -> sha of content placed there
        self.rows: list[dict] = []
        self.conflicts: list[str] = []

    def put_file(self, src: Path, dest_rel: str, role: str, note: str = "") -> None:
        digest = sha256(src)
        self._record(digest, dest_rel, role, src, note, lambda d: shutil.copy2(src, d))

    def put_bytes(self, data: bytes, dest_rel: str, role: str, src: Path, note: str) -> None:
        digest = hashlib.sha256(data).hexdigest()
        self._record(digest, dest_rel, role, src, note, lambda d: d.write_bytes(data))

    def _record(self, digest, dest_rel, role, src, note, write):
        dest = OUT / dest_rel
        prev = self.dest_hash.get(dest)
        if prev is not None and prev != digest:
            self.conflicts.append(f"{dest_rel}\n    <- {src.relative_to(RESULTS)}")
            return
        if prev is None:
            self.dest_hash[dest] = digest
            if not self.dry:
                dest.parent.mkdir(parents=True, exist_ok=True)
                write(dest)
        self.rows.append({"path": dest_rel, "role": role, "sha256": digest,
                          "source": str(src.relative_to(RESULTS)), "note": note})


# ───────────────────────── robot results ─────────────────────────

def grid_results(tree: Tree) -> None:
    """Sep 29–30 grid: split every result file into one CSV per (cell, seed)."""
    for f in sorted((INCOMING / "grid_out").glob("s*/results/*.csv")):
        if f.name.endswith((".bad.csv", ".partial_oom.csv")):
            continue
        seed = f.parts[-3][1:]
        m = re.match(r"llm_(res224|res32|april)_", f.name)
        llm_set = m.group(1) if m else None
        text = f.read_text().splitlines(keepends=True)
        header = text[0]
        for (cs, isrc, fam), rs in read_cells(f).items():
            fields = {"rule": "sparse", "concepts": CONCEPTS[cs], "arch": fam, "isrc": isrc,
                      "strategy": "upto", "seed": seed}
            if fam == "probcbm":
                fields["mode"] = "joint"
            if cs in ("machine_annotation", "llm_concepts", "clip_concepts"):
                fields["enc"] = "hard"
            if isrc == "llm":
                if llm_set == "april":
                    fields.update({"llm": "gemini-april-cache", "img": "32px", "questions": "v1-attribute-names"})
                else:
                    fields.update({"llm": "gemini-2.5-flash-lite", "img": llm_set.replace("res", "") + "px",
                                   "questions": "v2-value-explicit"})
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
            tree.put_bytes(buf.getvalue().encode(), "robot/grid/" + name("robot", fields, "results", ".csv"),
                           "grid", f, f"rows of cell ({cs}, {isrc}, {fam}) from {f.name}")


JULY_ROBOT_PATTERNS = [
    # (regex on file name, folder under robot/archive, extra fields from the run scripts)
    (r"RES_s(\d+)_(\w+?)_(ideal|subconcept)_f12\.csv", "july-zoo-perfect", {}),
    (r"RES_s(\d+)_(\w+?)_(ideal|subconcept)_f4_(hard_revealed|hard|koh595)\.csv", "july-machine-expert-exactk", {"strategy": "exact"}),
    (r"RES_s(\d+)_probcbm_(foot_subtypes|ground_truth|subconcept)_JOINT\.csv", "july-probcbm-joint-snapshots", {}),
    (r"RES_s(\d+)_probcbm_subconcept_ABL\.csv", "july-probcbm-ablation-p0", {}),
    (r"HARVEST_robot_image_stochastic_(ideal|subconcept)_(\w+?)_seed(\d+)_(\w+?)_(perfect|expert|llm)\.csv", "july-harvest-snapshots", {}),
    (r"robot_image_stochastic_(ideal|subconcept)_(\w+?)_seed(\d+)_results\.csv", "july-pipeline-outputs", {}),
]


def july_robot_results(tree: Tree, src: Path) -> bool:
    for rx, folder, extra in JULY_ROBOT_PATTERNS:
        m = re.fullmatch(rx, src.name)
        if not m:
            continue
        cells = read_cells(src)
        if len(cells) != 1:
            tree.put_file(src, f"robot/archive/{folder}/unlabeled__{src.name}", "archive", "multi-cell file")
            return True
        (cs, isrc, fam), rs = next(iter(cells.items()))
        seed = next(g for g in m.groups() if g and g.isdigit() and len(g) == 4)
        fields = {"rule": "sparse", "concepts": CONCEPTS.get(cs, cs), "arch": fam, "isrc": isrc,
                  "strategy": extra.get("strategy", "upto"), "seed": seed, "saved": saved(src)}
        if "f4_" in src.name and cs == "machine_annotation":
            fields["enc"] = ENC[m.group(4)]
        if fam == "probcbm":
            fields["mode"] = probcbm_mode(src, seed, cs)
        if "robots_full_backup" in src.parts:  # older July vintage: machine runs carry 13 concepts
            folder = "july-older-vintage-13-concepts"
        tree.put_file(src, f"robot/archive/{folder}/" + name("robot", fields, "results", ".csv"), "archive",
                      "superseded by robot/grid (Sep 29-30)")
        return True
    return False


def probcbm_mode(src: Path, seed: str, cs: str) -> str:
    """Training mode of a July ProbCBM result, from the run scripts and logs (see memory note)."""
    n = src.name
    if "ABL" in n or (n.startswith("robot_image_stochastic_subconcept_probcbm") and seed in {"1014", "1015", "1016", "1017", "1019"}
                      and datetime.fromtimestamp(src.stat().st_mtime) >= datetime(2026, 7, 30, 6, 30)):
        return "joint-p0"
    if "_f12" in n or "_f4_" in n:
        return "independent"
    if cs == "clip_concepts":
        return "joint"  # final_cells.sh:25 / probcbm_mountain.sh clip loop use --training-mode joint
    if n.startswith("robot_image_stochastic_ideal_probcbm") and seed in {"1015", "1016", "1017", "1018"}:
        return "joint"  # final_cells.sh:20 joint_ideal cells trained (logs show epochs)
    return "unknown"


def balanced_rule(tree: Tree, src: Path) -> bool:
    m = re.fullmatch(r"robot_(ideal|subconcept)_seed(\d+)_([0-9a-f]{8})_results\.csv", src.name)
    if not m:
        return False
    preset, seed, cfg = m.groups()
    rows = list(csv.DictReader(src.open()))
    models = sorted({r.get("model", "") for r in rows})
    fields = {"concepts": PRESET[preset], "arch": "-and-".join(models) or None, "seed": seed, "cfg": cfg}
    if BALANCED_PINNED[preset].get(seed) == cfg:
        fields["rule"] = "balanced"
        tree.put_file(src, "robot/balanced_rule/" + name("robot", fields, "results", ".csv"), "cited", "Figure 1a")
    else:
        tree.put_file(src, "robot/exploratory/collect_sweeps/" + name("robot", fields, "collect", ".csv"),
                      "exploratory", "rule not recorded; cfg = config hash")
    return True


def alignment(tree: Tree, src: Path) -> bool:
    m = re.fullmatch(r"robot_image_stochastic_(ideal|subconcept)_seed(\d+)_alignment\.json", src.name)
    if not m:
        return False
    fields = {"rule": "sparse", "concepts": PRESET[m.group(1)], "arch": "cbm", "seed": m.group(2)}
    tree.put_file(src, "robot/alignment/" + name("robot", fields, "alignment", ".json"), "cited", "alignment figure")
    return True


# ───────────────────────── sudoku ─────────────────────────

def sudoku(tree: Tree, src: Path) -> bool:
    n = src.name
    m = re.fullmatch(r"CELL(2?)_(iv|co)_s(\d+)_r(\d+)_(\w+?)_t(\d+)\.csv", n)
    if m:
        refit, what, seed, res, arch, tau = m.groups()
        base = {"res": f"{res}px", "tau": TAU[tau], "seed": seed,
                "threshold": "per-budget" if refit else "fit-at-k0"}
        if what == "iv":
            fields = {"arch": ARCH[arch], **base}
            folder = "sudoku/cells" if refit else "sudoku/archive/threshold_fit_at_k0"
            tree.put_file(src, f"{folder}/" + name("sudoku", fields, "interventions", ".csv"),
                          "cited" if refit else "archive", "")
        else:  # selective files hold only the CBM (cs) and DNN rows, whatever arch the name says
            models = sorted({ARCH.get(r["model"], r["model"]) for r in csv.DictReader(src.open())})
            fields = {"arch": "-and-".join(models), **base}
            # files differ by which family's pipeline run wrote them (identical ones still dedupe by hash)
            fields["by"] = f"{ARCH[arch]}-run"
            folder = "sudoku/selective_by_tau" if refit else "sudoku/archive/threshold_fit_at_k0"
            tree.put_file(src, f"{folder}/" + name("sudoku", fields, "selective", ".csv"),
                          "exploratory" if refit else "archive", "")
        return True
    m = re.fullmatch(r"sudoku_selective_tabular_n\d+_mc\d+_px(\d+)_seed(\d+)\.csv", n)
    if m:
        res, seed = m.groups()
        models = sorted({ARCH.get(r["model"], r["model"]) for r in csv.DictReader(src.open())})
        fields = {"arch": "-and-".join(models), "res": f"{res}px", "seed": seed}
        if "refit_sel" in src.parts:
            fields["threshold"] = "per-budget"
            tree.put_file(src, "sudoku/selective/" + name("sudoku", fields, "selective-all-tau", ".csv"), "cited", "Figure 1b")
        else:
            fields["threshold"] = "fit-at-k0"
            tree.put_file(src, "sudoku/archive/threshold_fit_at_k0/" + name("sudoku", fields, "selective-all-tau", ".csv"), "archive", "")
        return True
    m = re.fullmatch(r"sudoku_seed(\d+)_([0-9a-f]{8})_results\.csv", n)
    if m:
        rows = list(csv.DictReader(src.open()))
        models = sorted({ARCH.get(r["model"], r["model"]) for r in rows})
        taus = sorted({r["target_accuracy"] for r in rows if r.get("target_accuracy")})
        fields = {"arch": "-and-".join(models), "tau": "-".join(taus), "seed": m.group(1), "cfg": m.group(2)}
        tree.put_file(src, "sudoku/exploratory/collect_sweeps/" + name("sudoku", fields, "collect", ".csv"),
                      "exploratory", "resolution not recorded; cfg = config hash")
        return True
    m = re.fullmatch(r"sudoku_(?:(\w+?)_)?interventions_tabular_n\d+_mc\d+(?:_px(\d+))?(?:_seed(\d+))?\.csv", n)
    if m:
        arch, res, seed = m.groups()
        fields = {"arch": ARCH.get(arch or "cs", arch), "res": f"{res}px" if res else None, "seed": seed, "saved": saved(src)}
        tree.put_file(src, "sudoku/exploratory/pipeline_last_output/" + name("sudoku", fields, "interventions", ".csv"),
                      "exploratory", "last pipeline write at this path; tau not recorded")
        return True
    return False


# ───────────────────────── models and data ─────────────────────────

GRID_JULY_REUSED = re.compile(r"robot_image_stochastic_4_(ideal|subconcept)_(cbm|cem|ecbm)_seed(\d+)\.model")


def model_fields(src: Path) -> tuple[dict, str] | None:
    """Fields and destination folder for a model file, or None if not a model."""
    n = src.name
    m = re.fullmatch(r"sudoku_(\w+?)(_independent)?_tabular_n\d+_mc\d+(?:_px(\d+))?(?:_seed(\d+))?\.model", n)
    if m:
        arch, indep, res, seed = m.groups()
        f = {"arch": ARCH.get(arch, arch), "res": f"{res}px" if res else None, "seed": seed}
        if indep:
            f["mode"] = "independent"
        if not seed:
            f["saved"] = saved(src)
        return f, "models/sudoku" if seed else "models/sudoku/exploratory_unseeded"
    m = re.fullmatch(r"robot_image_stochastic_4_(ideal|subconcept)_(\w+?)(_independent)?(?:_(machine|llm|clip|subjective))?(?:_seed(\d+))?\.model", n)
    if not m:
        return None
    preset, arch, indep, csrc, seed = m.groups()
    f = {"concepts": csrc or PRESET[preset], "arch": ARCH.get(arch, arch), "seed": seed}
    if csrc == "subjective":
        f["concepts"] = PRESET[preset] + "-noisy"
    in_grid = "grid_models" in src.parts
    if in_grid:
        f["rule"] = "sparse"
        july = GRID_JULY_REUSED.fullmatch(n) is not None
        f["trained"] = "2026-07" if july else "2026-09"
        if arch == "probcbm":
            f["mode"] = "joint"
        return f, "models/robot/grid"
    f["saved"] = saved(src)
    if indep:
        f["mode"] = "independent"
    if not seed:
        return f, "models/robot/exploratory_unseeded"
    if "robots_full_backup" in src.parts:
        return f, "models/robot/archive/july-older-vintage"
    if arch == "probcbm" and csrc is None:
        mt = datetime.fromtimestamp(src.stat().st_mtime)
        f["mode"] = "joint-p0" if (preset == "subconcept" and mt >= datetime(2026, 7, 30, 6, 30)) else "independent"
    elif arch == "probcbm" and csrc == "clip":
        f["mode"] = "joint"
    return f, "models/robot/archive/july"


def model(tree: Tree, src: Path) -> bool:
    got = model_fields(src)
    if got is None:
        return False
    f, folder = got
    tree.put_file(src, f"{folder}/" + name(src.name.split("_")[0] if src.name.startswith("sudoku") else "robot", f, "model", ".pt"),
                  "model", "")
    return True


def dataset(tree: Tree, src: Path) -> bool:
    m = re.fullmatch(r"robot_image_4_(ideal|subconcept)_seed(\d+)\.data", src.name)
    if not m:
        return False
    f = {"rule": "sparse", "concepts": PRESET[m.group(1)], "seed": m.group(2)}
    tree.put_file(src, "robot/datasets/" + name("robot", f, "dataset", ".data"), "data", "July dataset used by the grid")
    return True


def llm_cache(tree: Tree, src: Path) -> bool:
    m = re.fullmatch(r"llm_interventions_([0-9a-f]{40})_([0-9a-f]{40})\.jsonl", src.name)
    if not m:
        return False
    cset, seed = CACHE_CONCEPTS[m.group(1)[:8]], TEST_SET_SEED[m.group(2)[:8]]
    group = src.parent.name
    if group == "april":
        f = {"concepts": cset, "llm": "gemini-april-cache", "img": "32px", "questions": "v1-attribute-names", "seed": seed}
    else:
        f = {"concepts": cset, "llm": "gemini-2.5-flash-lite", "img": group.replace("res", "") + "px",
             "questions": "v2-value-explicit", "seed": seed}
    tree.put_file(src, "robot/llm_caches/" + name("robot", f, "llm-votes", ".jsonl"), "cache",
                  f"pipeline file name: {src.name}")
    return True


def leftovers(tree: Tree, src: Path) -> bool:
    """Seedless April-era outputs and non-benchmark files: named from verifiable fields where possible."""
    n = src.name
    m = re.fullmatch(r"robot_image_stochastic_(ideal|subconcept)_(cbm|cem|probcbm|ecbm)(_independent)?(_kmax)?_results\.csv", n)
    if m:
        cells = read_cells(src)
        (cs, isrc, fam), _ = next(iter(cells.items())) if len(cells) == 1 else ((None, None, m.group(2)), None)
        f = {"concepts": CONCEPTS.get(cs, PRESET[m.group(1)]), "arch": fam, "isrc": isrc,
             "mode": "independent" if m.group(3) else None, "saved": saved(src)}
        kind = "results-kmax" if m.group(4) else "results"
        tree.put_file(src, "robot/exploratory/unseeded/" + name("robot", f, kind, ".csv"), "exploratory", "no seed in run")
        return True
    m = re.fullmatch(r"robot_image_stochastic_(ideal|subconcept)_ecbm(?:_seed(\d+))?_interpretation\.csv", n)
    if m:
        f = {"rule": "sparse", "concepts": PRESET[m.group(1)], "arch": "ecbm", "seed": m.group(2), "saved": saved(src)}
        tree.put_file(src, "robot/exploratory/ecbm_interpretation/" + name("robot", f, "interpretation", ".csv"), "exploratory", "")
        return True
    m = re.fullmatch(r"sudoku_(?:(\w+?)(_independent)?_)?selective_tabular_n\d+_mc\d+(?:_px(\d+))?\.csv", n)
    if m:
        arch, indep, res = m.groups()
        f = {"arch": ARCH.get(arch or "cs", arch), "mode": "independent" if indep else None,
             "res": f"{res}px" if res else None, "saved": saved(src)}
        tree.put_file(src, "sudoku/exploratory/unseeded/" + name("sudoku", f, "selective", ".csv"), "exploratory", "no seed in run")
        return True
    if n in ("pistachio_realworld_results.csv", "ppe_automation_results.csv", "crossover_sweep.csv"):
        tree.put_file(src, f"other/{n}", "other", "not a robot/sudoku benchmark result")
        return True
    if src.suffix == ".pdf":
        tree.put_file(src, f"figures/{n}", "figure", "")
        return True
    if n == "_MYRETRAIN_lfcbm_machine_seed1014.model":
        f = {"concepts": "machine", "arch": "lfcbm", "seed": "1014", "saved": saved(src)}
        tree.put_file(src, "models/robot/exploratory_retrain_test/" + name("robot", f, "model", ".pt"), "model", "manual retrain test")
        return True
    return False


# ───────────────────────── driver ─────────────────────────

def sources() -> list[Path]:
    roots = [RAW, CAMERA, INCOMING / "grid_models", INCOMING / "dsmlp_results", INCOMING / "llm_caches"]
    out = []
    for r in roots:
        out += [p for p in r.rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")]
    return sorted(out)


def classify(tree: Tree, src: Path) -> None:
    if src.suffix == ".log" and "llm_caches" in src.parts:
        tree.put_file(src, f"robot/llm_caches/build_logs/{src.parent.name}__{src.name}", "log", "")
        return
    for handler in (llm_cache, dataset, model, alignment, balanced_rule, sudoku, leftovers):
        if handler(tree, src):
            return
    if src.suffix == ".csv" and july_robot_results(tree, src):
        return
    if src.suffix in (".py", ".sh"):
        tree.put_file(src, f"scripts/aggregation/{src.name}", "script", "")
        return
    if src.name == "leakage.log":
        tree.put_file(src, "recovered/leakage__diagnose_leakage-stdout.log", "recovered", "Table::Leakage source")
        return
    rel = src.relative_to(RESULTS)
    tree.put_file(src, f"unlabeled/{rel.parts[0]}/{'/'.join(rel.parts[1:])}", "unlabeled",
                  "no verifiable fields; original name kept")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.dry_run:
        sys.exit(f"{OUT} exists; remove it first")
    tree = Tree(dry=args.dry_run)

    grid_results(tree)
    for f in sorted((INCOMING / "grid_out").rglob("*")):
        if f.is_file() and f.suffix in (".log", ".txt") and f.name != "grid_table.txt":
            rel = f.relative_to(INCOMING / "grid_out")
            tree.put_file(f, "robot/grid/run_logs/" + "/".join(rel.parts), "log", "Sep 29-30 grid run log")
    for f in sorted((INCOMING / "grid_run_scripts").glob("*")):
        tree.put_file(f, f"robot/grid/run_scripts/{f.name}", "script", "script that produced the Sep 29-30 grid")
    for src in sources():
        classify(tree, src)
    for d, sub in ((INCOMING / "robot_images_224", "robot_224px"), (INCOMING / "robot_images_32_july", "robot_32px")):
        n = len(list(d.glob("*.png")))
        if not args.dry_run:
            shutil.copytree(d, OUT / "images" / sub)
        tree.rows.append({"path": f"images/{sub}/", "role": "images", "sha256": "", "source": str(d.relative_to(RESULTS)),
                          "note": f"{n} test-robot renders (seeds 1014-1017), pipeline file names"})
    for extra in ("RECOVERED", "PENDING_RERUNS.md"):
        p = OLD_PAPER / extra
        if p.is_dir():
            for f in p.iterdir():
                tree.put_file(f, f"recovered/{f.name}", "recovered", "values recovered from transcripts")
        elif p.exists():
            tree.put_file(p, "PENDING_RERUNS.md", "doc", "")

    if tree.conflicts:
        print("NAME CONFLICTS (different content, same name):")
        print("\n".join(sorted(set(tree.conflicts))[:40]))
        sys.exit(1)
    if not args.dry_run:
        with (OUT / "INDEX.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "role", "sha256", "source", "note"])
            w.writeheader()
            w.writerows(tree.rows)
    placed = len(tree.dest_hash)
    by_top = defaultdict(int)
    for d in tree.dest_hash:
        by_top["/".join(d.relative_to(OUT).parts[:2])] += 1
    for k in sorted(by_top):
        print(f"{by_top[k]:6d}  {k}")
    print(f"{placed} files placed from {len(tree.rows)} source records; conflicts 0")


if __name__ == "__main__":
    main()
