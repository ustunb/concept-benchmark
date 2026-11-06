"""Write the robot text-modality table: a Glorp and a Drent, each with a caption from every text corpus and its image.

Generates the text dataset at each template complexity (low: simple templates, medium: semi-structured templates,
high: LLM-generated captions) with the text pipeline's seed, takes the captions of two fixed robots, and renders their
images at 600 px with the color scheme they have in the image catalog. Semi-structured captions name a color drawn at random, so it
need not match the image.

    python scripts/paper/make_robot_text_table.py --out ../concept-benchmark-paper/tables/robot_modalities.tex \
        --figures-dir ../concept-benchmark-paper/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import REPO  # noqa: F401  (puts the repository root on the import path)
from concept_benchmark.config import ROBOT_CONCEPTS
from concept_benchmark.robots import DatasetGenerator
from concept_benchmark.synthetic.robot.catalog import get_robot_catalog_df
from concept_benchmark.synthetic.robot.draw import draw_robot
from concept_benchmark.synthetic.robot.text.catalog import enumerate_robot_concepts

SEED = 1337  # the text pipeline's seed
LEVELS = [
    ("low", "Simple templates"),
    ("medium", "Semi-structured"),
    ("high", "LLM-generated"),
]
ROBOTS = [
    ("Glorp", 6448),
    ("Drent", 1415),
]  # rows of the text catalog at SEED, drawn at random among captions without the corpus typos
IMAGE_PX = 600
LATEX_ESCAPES = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$"}


def escape_latex(text: str) -> str:
    """Caption text with LaTeX special characters escaped and typographic quotes kept."""
    return "".join(LATEX_ESCAPES.get(ch, ch) for ch in text)


def captions_by_level() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Level -> (captions, labels, catalog row of each caption)."""
    out = {}
    for level, _ in LEVELS:
        ds = DatasetGenerator(
            seed=SEED, data_type="text", template_complexity=level
        ).generate()
        out[level] = (
            np.asarray(ds.X),
            np.asarray(ds.y),
            np.asarray(ds.meta["row_index"]),
        )
    return out


def color_scheme_of(features: dict[str, str]) -> int:
    """Color scheme index of the robot with these body parts in the image catalog."""
    catalog = get_robot_catalog_df(ROBOT_CONCEPTS)
    match = catalog
    for name, value in features.items():
        match = match[match[name] == value]
    if len(match) != 1:
        raise SystemExit(
            f"{features}: {len(match)} robots in the image catalog, expected 1"
        )
    return int(match["color_scheme"].iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures-dir", type=Path, required=True)
    args = ap.parse_args()

    captions = captions_by_level()
    text_catalog = enumerate_robot_concepts(concepts=ROBOT_CONCEPTS, seed=SEED)
    # label and image side by side, captions vertically centered; the LLM captions are the longest
    ragged = r">{\raggedright\arraybackslash}m"
    columns = f"l >{{\\centering\\arraybackslash}}m{{1.3in}} {ragged}{{1.25in}} {ragged}{{1.35in}} {ragged}{{2.2in}}"
    lines = [
        r"\small",
        r"\renewcommand{\arraystretch}{1.2}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        r"\multicolumn{2}{@{}l}{\textbf{Robot}} & "
        + " & ".join(rf"\textbf{{{name}}}" for _, name in LEVELS)
        + r" \\",
        r"\midrule",
    ]
    for label, row in ROBOTS:
        features = {name: str(text_catalog.loc[row, name]) for name in ROBOT_CONCEPTS}
        texts = []
        for level, _ in LEVELS:
            X, y, rows = captions[level]
            (i,) = np.flatnonzero(rows == row)
            if (y[i] == 1) != (label == "Glorp"):
                raise SystemExit(f"row {row} is labeled {y[i]}, expected a {label}")
            texts.append(f"``{escape_latex(str(X[i]))}''")
        png = args.figures_dir / f"robot_text_{label.lower()}.png"
        draw_robot(
            filetype="png",
            width=IMAGE_PX,
            height=IMAGE_PX,
            color_scheme=color_scheme_of(features),
            **features,
        ).export(str(png))
        image = rf"\includegraphics[width=1.25in]{{figures/{png.name}}}"
        lines.append(f"{label} & {image} & " + " & ".join(texts) + r" \\")
        lines.append(r"\midrule" if label != ROBOTS[-1][0] else r"\bottomrule")
        print(f"{label}: row {row} {features}")
    lines.append(r"\end{tabular}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} and {len(ROBOTS)} images in {args.figures_dir}")


if __name__ == "__main__":
    main()
