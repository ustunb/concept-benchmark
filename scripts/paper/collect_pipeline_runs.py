"""Copy the output of the pipelines into the layout that the paper scripts read.

The pipelines write one results file per model family (`robot_image_stochastic_subconcept_cbm_seed1014_results.csv`);
the paper scripts read one file per cell (concept source x intervention source) and seed, named after its content.
This script splits and renames a folder of pipeline output, so that the tables, figures and tests can be rebuilt
from your own runs:

    python scripts/robot_pipeline.py --seed 1014 --concept-preset foot_subtypes --budgets 1 3 max
    python scripts/paper/collect_pipeline_runs.py --runs results --results-root my_results
    python scripts/paper/make_robot_big_table.py --results-root my_results --out table.tex

The pipelines do not record every setting of a run in their output, so pass the ones that differ from the defaults:
`--strategy`, `--encoding` (the `--intervention-encoding` of the run), `--probcbm-mode`, `--llm-tag`, and
`--target-accuracy` for sudoku runs. Run the script once per group of runs that share these settings.
Existing files are never overwritten: a file that is already there with other content is an error.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

from _common import BALANCED_TAG, PaperResults

CONCEPTS = {
    "ground_truth": "true",
    "human_concepts": "human",
    "noisy_human_concepts": "noisy-human",
    "machine_annotation": "machine",
    "llm_concepts": "llm",
    "clip_concepts": "clip",
}
LABEL_FREE = {"machine_annotation", "llm_concepts", "clip_concepts"}
STRATEGIES = {"up_to_k": "upto", "exactly_k": "exact"}
ENCODINGS = {"binary": "hard", "percentile": "koh595", "binary_revealed": "hardrev"}
PAPER_LLM_TAG = "llm-gemini-2.5-flash-lite__img-224px__questions-v2-value-explicit"

ROBOT_CELLS = re.compile(
    r"robot_image_(?:stochastic|deterministic)_(?P<variant>ideal|subconcept)(?P<sparse>_sparse)?"
    r"_(?P<family>cbm|cem|probcbm|ecbm)_seed(?P<seed>\d+)_results\.csv"
)
ROBOT_SUMMARY = re.compile(
    r"robot_(?P<variant>ideal|subconcept)_seed(?P<seed>\d+)_[0-9a-f]{8}_results\.csv"
)
ROBOT_RECORDS = re.compile(
    r"(?P<family>[a-z]+)__(?P<concepts>[a-z_]+)__m(?P<n>\d+)_[0-9a-f]{8}__(?P<who>[a-z]+)(?P<accuracy>\d+)"
    r"__(?P<strategy>[a-z_]+)__(?P<encoding>[a-z_]+)__t[\d.]+__seed(?P<seed>\d+)__k(?P<budget>\d+)\.npz"
)
SUDOKU = re.compile(
    r"sudoku_(?P<key>[a-z_]+)_tabular_n\d+_mc\d+_px(?P<px>\d+)_seed(?P<seed>\d+)\.(?:csv|npz)"
)


def robot_tag(is_sparse: bool) -> str:
    return "robot__rule-sparse" if is_sparse else BALANCED_TAG


def arch_tag(family: str, probcbm_mode: str) -> str:
    return (
        f"arch-probcbm__mode-{probcbm_mode}"
        if family == "probcbm"
        else f"arch-{family}"
    )


def tau_tag(target_accuracy: float) -> str:
    """0.9 -> '0.90', 0.925 -> '0.925': the two or three decimals of the archived cells."""
    text = f"{target_accuracy:.3f}"
    return text[:-1] if text.endswith("0") else text


class Collector:
    def __init__(self, results: PaperResults, args: argparse.Namespace):
        self.results, self.args = results, args
        self.n_written, self.n_present, self.skipped = 0, 0, []

    def write(self, destination: Path, content: bytes) -> None:
        if destination.exists():
            if destination.read_bytes() != content:
                sys.exit(
                    f"{destination} exists with other content; use another --results-root"
                )
            self.n_present += 1
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        self.n_written += 1

    def robot_cells(self, path: Path, name: re.Match) -> None:
        """One file per (concept source, intervention source) of a family's results file."""
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            header, rows = reader.fieldnames, list(reader)
        cells = defaultdict(list)
        for row in rows:
            cells[(row["concept_source"], row["intervention_source"])].append(row)
        family = name["family"]
        for (concept_source, intervention_source), cell_rows in cells.items():
            fields = [
                robot_tag(bool(name["sparse"])),
                f"concepts-{CONCEPTS[concept_source]}",
                arch_tag(family, self.args.probcbm_mode),
                f"isrc-{intervention_source}",
            ]
            if intervention_source == "llm":
                fields.append(self.args.llm_tag)
            fields.append(f"strategy-{STRATEGIES[self.args.strategy]}")
            if concept_source in LABEL_FREE:
                is_label_free_cbm = family == "cbm"
                encoding = self.args.encoding if is_label_free_cbm else "binary"
                fields.append(f"enc-{ENCODINGS[encoding]}")
            fields += [f"seed-{name['seed']}", "results.csv"]
            text = io.StringIO(newline="")
            writer = csv.DictWriter(text, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(cell_rows)
            folder = (
                self.results.sparse_rule
                if name["sparse"]
                else self.results.balanced_rule
            )
            self.write(folder / "__".join(fields), text.getvalue().encode())

    def robot_summary(self, path: Path, name: re.Match, rules: set[bool]) -> None:
        """The collect stage's file with the DNN and the CBM, for the runs of one label rule."""
        models = {row["model"] for row in csv.DictReader(path.open(newline=""))}
        if "cbm" not in models:
            return  # the summaries of other families repeat the DNN row only
        if len(rules) != 1:
            self.skipped.append(f"{path.name}: cannot tell its label rule")
            return
        (is_sparse,) = rules
        concepts = "true" if name["variant"] == "ideal" else "human"
        folder = self.results.sparse_rule if is_sparse else self.results.balanced_rule
        self.write(
            folder
            / f"{robot_tag(is_sparse)}__concepts-{concepts}__arch-cbm-and-dnn__seed-{name['seed']}__results.csv",
            path.read_bytes(),
        )

    def robot_records(self, path: Path, name: re.Match) -> None:
        """Intervention records of `--dump-interventions` (balanced rule)."""
        if name["who"] in {"llm", "self"}:
            source = name["who"]
        else:
            source = "perfect" if name["accuracy"] == "100" else "expert"
        budget = "max" if name["budget"] == name["n"] else name["budget"]
        self.write(
            self.results.intervention_records
            / "__".join(
                [
                    BALANCED_TAG,
                    f"concepts-{CONCEPTS[name['concepts']]}",
                    arch_tag(name["family"], self.args.probcbm_mode),
                    f"isrc-{source}",
                    f"budget-{budget}",
                    f"seed-{name['seed']}",
                    "records.npz",
                ]
            ),
            path.read_bytes(),
        )

    def sudoku(self, path: Path, name: re.Match) -> None:
        key, stem = name["key"], f"res-{name['px']}px"
        seed = f"seed-{name['seed']}"
        if key.endswith("interventions"):
            if self.args.target_accuracy is None:
                sys.exit(f"{path.name}: pass the run's --target-accuracy")
            family = key.removesuffix("interventions").rstrip("_") or "cbm"
            destination = (
                self.results.sudoku_cells
                / f"sudoku__arch-{family}__{stem}__tau-{tau_tag(self.args.target_accuracy)}__threshold-per-budget__{seed}__interventions.csv"
            )
        elif key == "selective":
            destination = (
                self.results.sudoku_selective
                / f"sudoku__arch-cbm-and-dnn__{stem}__threshold-per-budget__{seed}__selective-all-tau.csv"
            )
        elif key.endswith("_confidence"):
            family = key.removesuffix("_confidence")
            family = "cbm" if family == "cs" else family
            destination = (
                self.results.sudoku_confidence
                / f"sudoku__arch-{family}__{stem}__{seed}__confidence.npz"
            )
        else:
            return
        self.write(destination, path.read_bytes())

    def collect(self, runs: Path) -> None:
        files = sorted(p for p in runs.rglob("*") if p.is_file())
        rules = defaultdict(set)  # (variant, seed) -> label rules of the runs found
        for path in files:
            if name := ROBOT_CELLS.fullmatch(path.name):
                rules[(name["variant"], name["seed"])].add(bool(name["sparse"]))
                self.robot_cells(path, name)
        for path in files:
            if name := ROBOT_SUMMARY.fullmatch(path.name):
                self.robot_summary(path, name, rules[(name["variant"], name["seed"])])
            elif name := ROBOT_RECORDS.fullmatch(path.name):
                self.robot_records(path, name)
            elif name := SUDOKU.fullmatch(path.name):
                self.sudoku(path, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Folder with pipeline output (searched recursively), e.g. results/.",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Folder to build; pass it to the other paper scripts as --results-root.",
    )
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="up_to_k")
    ap.add_argument(
        "--encoding",
        choices=sorted(ENCODINGS),
        default="binary",
        help="The --intervention-encoding of the runs (names the label-free CBM cells).",
    )
    ap.add_argument(
        "--probcbm-mode",
        default="joint",
        help="The --training-mode of the ProbCBM runs.",
    )
    ap.add_argument(
        "--llm-tag",
        default=PAPER_LLM_TAG,
        help="Name fields of the cells with LLM interventions (LLM, image size, question format).",
    )
    ap.add_argument(
        "--target-accuracy",
        type=float,
        default=None,
        help="The --target-accuracy of the sudoku runs.",
    )
    args = ap.parse_args()
    if not args.runs.is_dir():
        sys.exit(f"No folder at {args.runs}")
    collector = Collector(PaperResults(args.results_root), args)
    collector.collect(args.runs)
    for note in collector.skipped:
        print(f"skipped {note}")
    print(
        f"wrote {collector.n_written} files to {args.results_root} "
        f"({collector.n_present} were already there)"
    )


if __name__ == "__main__":
    main()
