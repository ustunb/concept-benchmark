"""The Python blocks of the README and the docs are valid code, and the sudoku training block runs."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAGES = [
    REPO / "README.md",
    REPO / "EXPERIMENTS.md",
    *sorted((REPO / "docs").glob("*.md")),
]


def python_blocks(page: Path) -> list[tuple[int, str]]:
    """(line number, code) of every ```python block of a page."""
    text = page.read_text()
    return [
        (text[: match.start()].count("\n") + 2, match.group(1))
        for match in re.finditer(r"```python\n(.*?)```", text, re.S)
    ]


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_python_blocks_compile(page):
    for line, code in python_blocks(page):
        compile(code, f"{page.name}:{line}", "exec")


def test_readme_sudoku_training_block_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the block saves its figure in the working folder
    (code,) = [
        code
        for _, code in python_blocks(REPO / "README.md")
        if "automation_table(" in code
    ]
    quick = (
        code.replace("epochs=100", "epochs=2")
        .replace("n_boards=1000", "n_boards=100")
        .replace(
            "dataset.train, dataset.val, seed=171)",
            "dataset.train, dataset.val, epochs=2, seed=171)",
        )
    )
    assert quick != code
    namespace = {}
    exec(compile(quick, "README.md", "exec"), namespace)
    results = namespace["results"]
    assert list(results["budget"]) == [0, 1, 3, 27]
    assert results["coverage_after"].between(0.0, 1.0).all()
    assert {"selective_accuracy_after", "total_concept_checks"} <= set(results.columns)


def _readme_blocks_containing(*needles: str) -> list[str]:
    blocks = [code for _, code in python_blocks(REPO / "README.md")]
    return [code for code in blocks if any(needle in code for needle in needles)]


@pytest.fixture
def tiny_cbm_namespace(tabular_factory):
    """`train`, `val`, `test`, `cd`, `fe` and `cbm` of a two-epoch CBM on a tiny tabular dataset: the names
    the README's intervention and alignment blocks expect."""
    from torch import nn

    from experiments.evaluate import train_cbm
    from experiments.models import (
        FrontEndModel,
    )  # imported by the block before `fe = FrontEndModel()`

    dataset, _ = tabular_factory(n=200, d=3, k=4, n_classes=2, with_cv=False)
    dataset.sample(test_size=0.25, val_size=0.25, stratify=dataset.y, seed=0)
    cbm = train_cbm(
        dataset.train,
        dataset.validation,
        detector=lambda: nn.Linear(3, 4),
        epochs=2,
        seed=0,
    )
    return {
        "train": dataset.train,
        "val": dataset.validation,
        "test": dataset.test,
        "cd": cbm.concept_detector,
        "fe": cbm.label_predictor,
        "cbm": cbm,
        "FrontEndModel": FrontEndModel,
    }


def test_readme_intervention_blocks_run(tiny_cbm_namespace, capsys):
    namespace = dict(tiny_cbm_namespace)
    blocks = _readme_blocks_containing(
        "fe.fit(train.C, train.y)",
        "concept_probs = cd.predict_proba(test)",
        "runner = ConceptInterventionRunner(model=cbm)",
        "class UncertaintyStrategy(InterventionStrategy):",
        "strategy=UncertaintyStrategy(),",
    )
    assert len(blocks) == 5
    for code in blocks:
        exec(compile(code, "README.md", "exec"), namespace)
    result = namespace["result"]
    assert result.mask.shape == (namespace["test"].n, 4)
    assert "Accuracy after" in capsys.readouterr().out


def test_readme_alignment_block_runs(tiny_cbm_namespace):
    pytest.importorskip("cvxpy")
    (code,) = _readme_blocks_containing("run_alignment(")
    namespace = dict(tiny_cbm_namespace)
    first_concept = namespace["train"].concepts[0]
    exec(
        compile(code.replace('"has_knees"', repr(first_concept)), "README.md", "exec"),
        namespace,
    )
    results = namespace["results"]
    assert {"original_accuracy", "aligned_accuracy", "accuracy_change"} <= set(results)
