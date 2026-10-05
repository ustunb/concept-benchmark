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


def test_readme_sudoku_training_block_runs():
    (code,) = [
        code
        for _, code in python_blocks(REPO / "README.md")
        if "GroupPoolingConceptSudokuCNN()" in code
    ]
    quick = code.replace('"epochs": 100', '"epochs": 2').replace(
        "n_boards=1000", "n_boards=100"
    )
    assert quick != code
    namespace = {}
    exec(compile(quick, "README.md", "exec"), namespace)
    assert 0.0 <= namespace["cov"] <= 1.0
