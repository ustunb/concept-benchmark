#!/usr/bin/env python
"""Format the robot benchmark CSV into paper-ready tables.

Reads ``results/robot_demo_results.csv`` (produced by ``collect_results()``)
and prints formatted tables matching those in the paper.  No model loading
— all data comes from the CSV.

Usage:
    python scripts/reproduce_robot_table.py                    # print all tables
    python scripts/reproduce_robot_table.py --table main       # just the main table
    python scripts/reproduce_robot_table.py --table alignment  # alignment table
    python scripts/reproduce_robot_table.py --collect           # re-collect then print
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from concept_benchmark.paths import results_dir

CSV_PATH = results_dir / "robot_demo_results.csv"


# ── Helpers ───────────────────────────────────────────────────────────

def _load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run the pipeline with --stages collect first.")
        sys.exit(1)
    return pd.read_csv(CSV_PATH)


def _pct(v) -> str:
    """Format a 0–1 float as a percentage with one decimal."""
    return f"{v * 100:.1f}"


def _lookup(df, dataset, model, budget=None):
    """Look up a single row and return it, or None."""
    mask = (df["dataset"] == dataset) & (df["model"] == model)
    if budget is not None:
        mask &= df["budget"] == budget
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _val(row, col, fmt="pct"):
    """Extract a formatted value from a row, or '—' if missing."""
    if row is None:
        return "—"
    v = row[col]
    if pd.isna(v) or v == "":
        return "—"
    if fmt == "pct":
        return _pct(float(v))
    if fmt == "signed_pct":
        fv = float(v)
        sign = "+" if fv >= 0 else ""
        return f"{sign}{fv * 100:.1f}"
    if fmt == "int":
        return str(int(float(v)))
    if fmt == "float2":
        return f"{float(v):.2f}"
    return str(v)


# ── Table printers ────────────────────────────────────────────────────

def print_main(df):
    """Main table (robot_cbm_main.tex): DNN + CBM k=0/1/3/max, ideal & subconcept."""
    print()
    print("=" * 100)
    print("  Table: Main Robot Results  (robot_cbm_main.tex)")
    print("=" * 100)
    header = f"{'Model':<28} {'Acc':>6} {'Gain':>7} {'Intv':>6} {'Avg':>5} {'Chgd':>6}  |  {'Acc':>6} {'Gain':>7} {'Intv':>6} {'Avg':>5} {'Chgd':>6}"
    print(f"{'':28} {'--- Ideal ---':^32}  |  {'--- Subconcept ---':^32}")
    print(header)
    print("-" * 100)

    for ds_label, ds_key in [("", "ideal"), ("", "subconcept")]:
        pass  # headers printed above

    def _row_line(label, dataset, model, budget=None):
        parts = []
        for ds in ["ideal", "subconcept"]:
            r = _lookup(df, ds, model, budget)
            parts.append(_val(r, "accuracy"))
            parts.append(_val(r, "gain", "signed_pct"))
            parts.append(_val(r, "predictions_intervened_on", "int"))
            parts.append(_val(r, "avg_concepts_per_sample", "float2"))
            parts.append(_val(r, "predictions_changed", "int"))
        ideal = parts[:5]
        sub = parts[5:]
        print(f"{label:<28} {ideal[0]:>6} {ideal[1]:>7} {ideal[2]:>6} {ideal[3]:>5} {ideal[4]:>6}  |  {sub[0]:>6} {sub[1]:>7} {sub[2]:>6} {sub[3]:>5} {sub[4]:>6}")

    _row_line("DNN", "ideal", "dnn")
    _row_line("CBM (k=0)", "ideal", "cbm", 0)
    _row_line("CBM (k=1)", "ideal", "cbm", 1)
    _row_line("CBM (k=3)", "ideal", "cbm", 3)
    # max budget
    cbm_rows = df[(df["model"] == "cbm") & (df["dataset"] == "ideal") & (df["budget"] > 3)]
    if not cbm_rows.empty:
        max_b = int(cbm_rows["budget"].max())
        _row_line(f"CBM (k={max_b}, max)", "ideal", "cbm", max_b)
    print("=" * 100)
    print()


def print_full(df):
    """Full table (robot_cbm.tex): adds aligned CBM + MCAR/MNAR rows."""
    print()
    print("=" * 110)
    print("  Table: Full Robot Results  (robot_cbm.tex)")
    print("=" * 110)

    header_cols = ["Acc", "Gain", "Intv", "Avg", "Chgd"]
    print(f"{'Dataset':<20} {'Model':<22} " + " ".join(f"{c:>6}" for c in header_cols))
    print("-" * 110)

    datasets = ["ideal", "subconcept", "ideal_mcar", "subconcept_mcar",
                 "ideal_mnar", "subconcept_mnar"]

    for ds in datasets:
        ds_rows = df[df["dataset"] == ds]
        if ds_rows.empty:
            continue

        first = True
        for model in ["dnn", "cbm", "aligned_cbm"]:
            model_rows = ds_rows[ds_rows["model"] == model].sort_values("budget")
            for _, r in model_rows.iterrows():
                budget = r["budget"]
                if model == "dnn":
                    label = "DNN"
                elif model == "cbm":
                    b = int(budget) if not pd.isna(budget) and budget != "" else 0
                    label = f"CBM (k={b})"
                else:
                    b = int(budget) if not pd.isna(budget) and budget != "" else 0
                    label = f"Aligned CBM (k={b})"

                ds_col = ds if first else ""
                first = False
                vals = [
                    _val(r, "accuracy"),
                    _val(r, "gain", "signed_pct"),
                    _val(r, "predictions_intervened_on", "int"),
                    _val(r, "avg_concepts_per_sample", "float2"),
                    _val(r, "predictions_changed", "int"),
                ]
                print(f"{ds_col:<20} {label:<22} " + " ".join(f"{v:>6}" for v in vals))
        print("-" * 110)
    print()


def print_alignment(df):
    """Alignment table (robot_cbm_alignment.tex): DNN, CBM, Aligned CBM with gain."""
    print()
    print("=" * 80)
    print("  Table: Alignment  (robot_cbm_alignment.tex)")
    print("=" * 80)
    print(f"{'Model':<28} {'Ideal Acc':>10} {'Ideal Gain':>11} {'Sub Acc':>10} {'Sub Gain':>11}")
    print("-" * 80)

    for model, budget, label in [
        ("dnn", None, "DNN"),
        ("cbm", 0, "CBM (k=0)"),
        ("aligned_cbm", 0, "Aligned CBM (k=0)"),
        ("aligned_cbm", 3, "Aligned CBM (k=3)"),
    ]:
        ideal = _lookup(df, "ideal", model, budget)
        sub = _lookup(df, "subconcept", model, budget)
        print(
            f"{label:<28} "
            f"{_val(ideal, 'accuracy'):>10} "
            f"{_val(ideal, 'gain', 'signed_pct'):>11} "
            f"{_val(sub, 'accuracy'):>10} "
            f"{_val(sub, 'gain', 'signed_pct'):>11}"
        )
    print("=" * 80)
    print()


def print_budget(df):
    """Budget table (robot_tab.tex): DNN, CBM k=0/1/3 with gain."""
    print()
    print("=" * 90)
    print("  Table: Budget Comparison  (robot_tab.tex)")
    print("=" * 90)
    print(f"{'Dataset':<20} {'DNN':>8} {'CBM k=0':>8} {'CBM k=1':>8} {'CBM k=3':>8} {'Gain k=3':>9}")
    print("-" * 90)

    datasets = ["ideal", "subconcept", "ideal_mcar", "subconcept_mcar",
                 "ideal_mnar", "subconcept_mnar"]

    for ds in datasets:
        dnn = _lookup(df, ds, "dnn")
        cbm0 = _lookup(df, ds, "cbm", 0)
        cbm1 = _lookup(df, ds, "cbm", 1)
        cbm3 = _lookup(df, ds, "cbm", 3)
        if dnn is None and cbm0 is None:
            continue
        print(
            f"{ds:<20} "
            f"{_val(dnn, 'accuracy'):>8} "
            f"{_val(cbm0, 'accuracy'):>8} "
            f"{_val(cbm1, 'accuracy'):>8} "
            f"{_val(cbm3, 'accuracy'):>8} "
            f"{_val(cbm3, 'gain', 'signed_pct'):>9}"
        )
    print("=" * 90)
    print()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Format robot benchmark CSV into paper tables.",
    )
    parser.add_argument(
        "--table",
        choices=["main", "full", "alignment", "budget", "all"],
        default="all",
        help="Which table to print (default: all).",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Re-run collect_results() before printing.",
    )
    args = parser.parse_args()

    if args.collect:
        from concept_benchmark.benchmarks.robot import collect_results
        print("Collecting results...")
        collect_results()
        print()

    df = _load()

    table = args.table
    if table in ("main", "all"):
        print_main(df)
    if table in ("full", "all"):
        print_full(df)
    if table in ("alignment", "all"):
        print_alignment(df)
    if table in ("budget", "all"):
        print_budget(df)


if __name__ == "__main__":
    main()
