"""Summarize the robot intervention grid from `results/paper/robot/grid/`.

Every file there is one (cell, seed); the cell is read from the file's own columns (concept_source,
intervention_source, model_family) and the LLM cache set from its name fields (`llm`, `img`).
Prints accuracy as mean ± SE over seeds for k = 0, 1, 3, max: res224 / res32 cover seeds 1015–1017,
april is seed 1014's April cache.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

CONCEPT_SOURCES = ["ground_truth", "human_concepts", "machine_annotation", "llm_concepts", "clip_concepts"]
ARCHS = ["cbm", "cem", "probcbm", "ecbm"]


def mean_se(values: list[float]) -> str:
    if not values:
        return "—"
    m = 100 * st.mean(values)
    se = 100 * st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else float("nan")
    return f"{m:.1f}±{se:.1f}" if len(values) > 1 else f"{m:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", type=Path, nargs="?", default=Path(__file__).resolve().parents[2] / "results/paper/robot/grid")
    args = ap.parse_args()

    cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds_seen: dict[tuple, set] = defaultdict(set)
    for f in sorted(args.grid.glob("*.csv")):
        fields = dict(kv.split("-", 1) for kv in f.stem.split("__")[1:-1])
        group = "-"
        if fields.get("isrc") == "llm":
            group = "april" if fields["llm"] == "gemini-april-cache" else "res" + fields["img"].removesuffix("px")
        rows = sorted(csv.DictReader(f.open()), key=lambda r: int(r["budget"]))
        cells_in_file = {(r["concept_source"], r["intervention_source"], r["model_family"]) for r in rows}
        if len(cells_in_file) != 1:
            raise SystemExit(f"{f.name}: expected one cell, found {cells_in_file}")
        cs, isrc, fam = cells_in_file.pop()
        key = (cs, isrc, fam, group)
        if fields["seed"] in seeds_seen[key]:
            raise SystemExit(f"duplicate cell {key} for seed {fields['seed']}")
        seeds_seen[key].add(fields["seed"])
        for label, r in zip(["k=0", "k=1", "k=3", "k=max"], rows):
            cells[key][label].append(float(r["accuracy"]))

    groups = [("perfect", "-"), ("expert", "-"), ("llm", "res224"), ("llm", "res32"), ("llm", "april")]
    print(f"{'concept source':20} {'interventions':16} {'arch':8} {'n':>2}  {'k=0':>10} {'k=1':>10} {'k=3':>10} {'k=max':>10}")
    missing = 0
    for cs in CONCEPT_SOURCES:
        for isrc, grp in groups:
            label = isrc if grp == "-" else f"llm ({grp})"
            for fam in ARCHS:
                key = (cs, isrc, fam, grp)
                if key not in cells:
                    print(f"{cs:20} {label:16} {fam:8} {'':>2}  MISSING")
                    missing += 1
                    continue
                c = cells[key]
                n = len(seeds_seen[key])
                print(f"{cs:20} {label:16} {fam:8} {n:>2}  " + " ".join(f"{mean_se(c[k]):>10}" for k in ["k=0", "k=1", "k=3", "k=max"]))
    print(f"\nmissing cells: {missing}")


if __name__ == "__main__":
    main()
