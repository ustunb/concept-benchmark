"""Compare a re-evaluated robot grid against the installed one, cell by cell.

Old cells: `results/paper/robot/grid/*.csv` (one file per cell and seed, fields in the name).
New cells: `<new>/s<seed>/results/<run>.csv` as written by the re-evaluation worker (one file per run,
possibly several cells). Rows are matched on (seed, concept_source, intervention_source, model_family,
LLM cache set, budget).

Expected: k=0 identical everywhere; k=1/3 identical except ProbCBM and machine-annotated cells;
k=max may change. Anything else is reported as unexpected, and the exit code is 1.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

N_SEEDS_EXPECTED = 4


def llm_set_from_old(fields: dict[str, str]) -> str:
    if fields.get("isrc") != "llm":
        return "-"
    return "april" if fields["llm"] == "gemini-april-cache" else "res" + fields["img"].removesuffix("px")


def read_old(grid: Path) -> dict[tuple, list[str]]:
    rows = {}
    for f in grid.glob("*.csv"):
        fields = dict(kv.split("-", 1) for kv in f.stem.split("__")[1:-1])
        for r in csv.DictReader(f.open()):
            key = (fields["seed"], r["concept_source"], r["intervention_source"], r["model_family"],
                   llm_set_from_old(fields))
            rows.setdefault(key, []).append((int(r["budget"]), r["accuracy"]))
    return {k: [a for _, a in sorted(v)] for k, v in rows.items()}


def read_new(root: Path) -> tuple[dict[tuple, list[str]], list[Path]]:
    rows, bad = {}, []
    for f in root.glob("s*/results/*.csv"):
        if f.name.endswith(".bad.csv"):
            bad.append(f)
            continue
        seed = f.parent.parent.name.removeprefix("s")
        llm_set = f.stem.split("_")[1] if f.stem.startswith("llm_") else "-"
        for r in csv.DictReader(f.open()):
            key = (seed, r["concept_source"], r["intervention_source"], r["model_family"], llm_set)
            rows.setdefault(key, []).append((int(r["budget"]), r["accuracy"]))
    return {k: [a for _, a in sorted(v)] for k, v in rows.items()}, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    args = ap.parse_args()

    old = read_old(args.old)
    new, bad = read_new(args.new)
    problems = [f"bad output file: {f}" for f in bad]
    problems += [f"missing in new: {k}" for k in sorted(set(old) - set(new))]
    problems += [f"not in old: {k}" for k in sorted(set(new) - set(old))]

    n_changed = {"k=0": 0, "k=1": 0, "k=3": 0, "k=max": 0}
    for key in sorted(set(old) & set(new)):
        o, n = old[key], new[key]
        if len(o) != 4 or len(n) != 4:
            problems.append(f"{key}: expected 4 budgets, old {len(o)} new {len(n)}")
            continue
        may_change_partial = key[3] == "probcbm" or key[1] == "machine_annotation"
        for label, a, b in zip(["k=0", "k=1", "k=3", "k=max"], o, n):
            if a == b:
                continue
            n_changed[label] += 1
            if label == "k=0" or (label in ("k=1", "k=3") and not may_change_partial):
                problems.append(f"{key} {label}: {a} -> {b}")

    print(f"cells compared: {len(set(old) & set(new))} (old {len(old)}, new {len(new)})")
    print("changed values:", n_changed)
    for p in problems:
        print("UNEXPECTED", p)
    print(f"unexpected differences: {len(problems)}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
