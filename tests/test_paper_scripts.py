"""The paper scripts start, read their results from the folder they are given, and accept collected pipeline runs."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
PAPER_SCRIPTS = REPO / "scripts" / "paper"
SCRIPT_NAMES = sorted(
    p.name for p in PAPER_SCRIPTS.glob("*.py") if p.name != "_common.py"
)
BALANCED_TAG = "robot__rule-balanced__sampling-skew0.30__elbows-weight2"
CELL_HEADER = (
    "budget,threshold,accuracy,concept_source,intervention_source,model_family"
)


def run_script(name: str, *arguments: str) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONPATH": str(REPO)}
    environment.pop("CONCEPT_BENCHMARK_PAPER_RESULTS", None)
    return subprocess.run(
        [sys.executable, str(PAPER_SCRIPTS / name), *map(str, arguments)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO,
    )


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_prints_its_help(name):
    result = run_script(name, "--help")
    assert result.returncode == 0, result.stderr[-2000:]


def test_missing_results_folder_names_the_download(tmp_path):
    result = run_script(
        "make_real_data_table.py",
        "--results-root",
        tmp_path / "absent",
        "--out",
        tmp_path / "table.tex",
    )
    assert result.returncode != 0
    assert "releases" in result.stderr and "--results-root" in result.stderr


def test_table_is_built_from_the_given_results_folder(tmp_path):
    folder = tmp_path / "results" / "real_datasets"
    folder.mkdir(parents=True)
    header = "dataset,concepts,seed,budget,accuracy\n"
    (folder / "derm7pt.csv").write_text(
        header
        + "derm7pt,clinician,0,0,0.5\nderm7pt,clinician,0,max,0.75\n"
        + "derm7pt,label_free,0,0,0.5\nderm7pt,label_free,0,max,0.25\n"
    )
    (folder / "cub.csv").write_text(
        header
        + "cub25,ground_truth,0,0,0.5\ncub25,ground_truth,0,max,0.75\n"
        + "cub25,label_free,0,0,0.5\ncub25,label_free,0,max,0.25\n"
    )
    out = tmp_path / "table.tex"
    result = run_script(
        "make_real_data_table.py", "--results-root", tmp_path / "results", "--out", out
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "human-annotated & 50.0\\% & 75.0\\% & $+25.0\\%$" in out.read_text()


@pytest.fixture
def pipeline_output(tmp_path):
    """A folder with the files that one robot run and one sudoku run write, under the pipelines' names."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "robot_image_stochastic_subconcept_cbm_seed7_results.csv").write_text(
        CELL_HEADER
        + "\n0,0.2,0.7,human_concepts,perfect,cbm\n12,0.2,0.9,human_concepts,perfect,cbm"
        + "\n0,0.2,0.7,human_concepts,llm,cbm\n0,0.2,0.6,clip_concepts,perfect,cbm\n"
    )
    (runs / "robot_image_stochastic_ideal_probcbm_seed7_results.csv").write_text(
        CELL_HEADER + "\n0,0.2,0.8,ground_truth,expert,probcbm\n"
    )
    (runs / "robot_subconcept_seed7_0123abcd_results.csv").write_text(
        "dataset,model,budget,accuracy\nsubconcept,dnn,,0.8\nsubconcept,cbm,0,0.7\n"
    )
    np.savez(
        runs
        / "cbm__human_concepts__m12_0123abcd__sim100__up_to_k__binary__t0.2__seed7__k12.npz",
        mask=np.zeros((1, 12), dtype=bool),
    )
    (runs / "sudoku_cem_interventions_tabular_n3_mc9_px18_seed5.csv").write_text(
        "budget,accuracy\n0,0.9\n"
    )
    (runs / "sudoku_selective_tabular_n3_mc9_px18_seed5.csv").write_text(
        "model,target_accuracy\ndnn,0.95\n"
    )
    return runs


def test_collected_runs_get_the_names_the_paper_scripts_read(pipeline_output, tmp_path):
    root = tmp_path / "collected"
    result = run_script(
        "collect_pipeline_runs.py",
        "--runs",
        pipeline_output,
        "--results-root",
        root,
        "--encoding",
        "percentile",
        "--target-accuracy",
        "0.9",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    llm = "llm-gemini-2.5-flash-lite__img-224px__questions-v2-value-explicit"
    expected = {
        f"robot/balanced_rule/{BALANCED_TAG}__concepts-human__arch-cbm__isrc-perfect__strategy-upto__seed-7__results.csv",
        f"robot/balanced_rule/{BALANCED_TAG}__concepts-human__arch-cbm__isrc-llm__{llm}__strategy-upto__seed-7__results.csv",
        f"robot/balanced_rule/{BALANCED_TAG}__concepts-clip__arch-cbm__isrc-perfect__strategy-upto__enc-koh595__seed-7__results.csv",
        f"robot/balanced_rule/{BALANCED_TAG}__concepts-true__arch-probcbm__mode-joint__isrc-expert__strategy-upto__seed-7__results.csv",
        f"robot/balanced_rule/{BALANCED_TAG}__concepts-human__arch-cbm-and-dnn__seed-7__results.csv",
        f"robot/balanced_rule/intervention_records/{BALANCED_TAG}__concepts-human__arch-cbm__isrc-perfect__budget-max__seed-7__records.npz",
        "sudoku/cells/sudoku__arch-cem__res-18px__tau-0.90__threshold-per-budget__seed-5__interventions.csv",
        "sudoku/selective/sudoku__arch-cbm-and-dnn__res-18px__threshold-per-budget__seed-5__selective-all-tau.csv",
    }
    collected = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert collected == expected


def test_collected_cell_keeps_only_its_own_rows(pipeline_output, tmp_path):
    root = tmp_path / "collected"
    run_script(
        "collect_pipeline_runs.py",
        "--runs",
        pipeline_output,
        "--results-root",
        root,
        "--target-accuracy",
        "0.9",
    )
    cell = (
        root
        / f"robot/balanced_rule/{BALANCED_TAG}__concepts-human__arch-cbm__isrc-perfect__strategy-upto__seed-7__results.csv"
    )
    assert cell.read_text() == (
        CELL_HEADER
        + "\n0,0.2,0.7,human_concepts,perfect,cbm\n12,0.2,0.9,human_concepts,perfect,cbm\n"
    )


def test_collecting_never_overwrites_other_content(pipeline_output, tmp_path):
    root = tmp_path / "collected"
    arguments = (
        "--runs",
        pipeline_output,
        "--results-root",
        root,
        "--target-accuracy",
        "0.9",
    )
    assert run_script("collect_pipeline_runs.py", *arguments).returncode == 0
    (pipeline_output / "sudoku_selective_tabular_n3_mc9_px18_seed5.csv").write_text(
        "model,target_accuracy\ndnn,0.99\n"
    )
    result = run_script("collect_pipeline_runs.py", *arguments)
    assert result.returncode != 0 and "exists with other content" in result.stderr
