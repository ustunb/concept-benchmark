#!/usr/bin/env python
"""Format the sudoku benchmark CSV into paper-ready tables.

Reads ``results/sudoku_demo_results.csv`` (produced by ``collect_results()``)
and prints formatted tables matching those in the paper.  No model loading
— all data comes from the CSV.

Usage:
    python scripts/reproduce_sudoku_table.py                    # print all tables
    python scripts/reproduce_sudoku_table.py --table main       # just the main table
    python scripts/reproduce_sudoku_table.py --table selective   # selective sweep
    python scripts/reproduce_sudoku_table.py --collect           # re-collect then print
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

CSV_PATH = results_dir / "sudoku_demo_results.csv"


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
    """Main table (sudoku_tab.tex): DNN + CS k=0/1/3/max per dataset."""
    print()
    print("=" * 100)
    print("  Table: Main Sudoku Results  (sudoku_tab.tex)")
    print("=" * 100)

    datasets = sorted(df["dataset"].unique())

    # Column headers: DNN, CS k=0, CS k=1, CS k=3, CS k=max
    header = (
        f"{'Dataset':<12} {'Metric':<12} "
        f"{'DNN':>8} {'CS k=0':>8} {'CS k=1':>8} {'CS k=3':>8} {'CS k=max':>8}"
    )
    print(header)
    print("-" * 100)

    for ds in datasets:
        # Determine max budget for this dataset
        cs_rows = df[(df["dataset"] == ds) & (df["model"] == "cs")]
        budgets_available = sorted(
            [int(b) for b in cs_rows["budget"].dropna() if b != "" and int(b) > 3]
        )
        max_budget = budgets_available[-1] if budgets_available else None

        dnn = _lookup(df, ds, "dnn")
        cs0 = _lookup(df, ds, "cs", 0)
        cs1 = _lookup(df, ds, "cs", 1)
        cs3 = _lookup(df, ds, "cs", 3)
        cs_max = _lookup(df, ds, "cs", max_budget) if max_budget else None

        # Selective Accuracy row
        vals = [
            _val(dnn, "selective_acc"),
            _val(cs0, "selective_acc"),
            _val(cs1, "selective_acc"),
            _val(cs3, "selective_acc"),
            _val(cs_max, "selective_acc"),
        ]
        print(f"{ds:<12} {'Sel. Acc':>12} " + " ".join(f"{v:>8}" for v in vals))

        # Coverage row
        vals = [
            _val(dnn, "selective_cov"),
            _val(cs0, "selective_cov"),
            _val(cs1, "selective_cov"),
            _val(cs3, "selective_cov"),
            _val(cs_max, "selective_cov"),
        ]
        print(f"{'':>12} {'Coverage':>12} " + " ".join(f"{v:>8}" for v in vals))

        # Avg concepts / predictions changed (only for k > 0)
        vals = [
            "—",
            "—",
            _val(cs1, "avg_concepts_per_sample", "float2"),
            _val(cs3, "avg_concepts_per_sample", "float2"),
            _val(cs_max, "avg_concepts_per_sample", "float2"),
        ]
        print(f"{'':>12} {'Avg Cpts':>12} " + " ".join(f"{v:>8}" for v in vals))
        print()

    print("=" * 100)
    print()


def print_selective(df):
    """Selective sweep: raw test acc + selective accuracy/coverage per dataset."""
    print()
    print("=" * 80)
    print("  Table: Selective Summary")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Model':<8} {'Raw Acc':>8} {'Sel. Acc':>9} {'Sel. Cov':>9}")
    print("-" * 80)

    datasets = sorted(df["dataset"].unique())
    for ds in datasets:
        for model in ["dnn", "cs", "aligned_cs"]:
            r = _lookup(df, ds, model, 0 if model in ("cs", "aligned_cs") else None)
            if r is None:
                continue
            print(
                f"{ds:<12} {model:<8} "
                f"{_val(r, 'raw_test_acc'):>8} "
                f"{_val(r, 'selective_acc'):>9} "
                f"{_val(r, 'selective_cov'):>9}"
            )
        print()

    print("=" * 80)
    print()


def print_full(df):
    """Full table: all (dataset, model, budget) rows."""
    print()
    print("=" * 110)
    print("  Table: Full Sudoku Results")
    print("=" * 110)
    cols = ["Sel. Acc", "Sel. Cov", "Raw Acc", "Intv", "Avg", "Chgd"]
    print(f"{'Dataset':<12} {'Model':<14} {'k':>3} " + " ".join(f"{c:>8}" for c in cols))
    print("-" * 110)

    datasets = sorted(df["dataset"].unique())
    for ds in datasets:
        ds_rows = df[df["dataset"] == ds]
        first = True
        for model in ["dnn", "cs", "aligned_cs"]:
            model_rows = ds_rows[ds_rows["model"] == model].sort_values("budget")
            for _, r in model_rows.iterrows():
                budget_str = str(int(r["budget"])) if pd.notna(r["budget"]) and r["budget"] != "" else "—"
                ds_col = ds if first else ""
                first = False
                vals = [
                    _val(r, "selective_acc"),
                    _val(r, "selective_cov"),
                    _val(r, "raw_test_acc"),
                    _val(r, "predictions_intervened_on", "int"),
                    _val(r, "avg_concepts_per_sample", "float2"),
                    _val(r, "predictions_changed", "int"),
                ]
                print(f"{ds_col:<12} {model:<14} {budget_str:>3} " + " ".join(f"{v:>8}" for v in vals))
        print("-" * 110)
    print()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Format sudoku benchmark CSV into paper tables.",
    )
    parser.add_argument(
        "--table",
        choices=["main", "selective", "full", "all"],
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
        from concept_benchmark.benchmarks.sudoku import collect_results
        print("Collecting results...")
        collect_results()
        print()

    df = _load()

    table = args.table
    if table in ("main", "all"):
        print_main(df)
    if table in ("selective", "all"):
        print_selective(df)
    if table in ("full", "all"):
        print_full(df)


if __name__ == "__main__":
    main()
