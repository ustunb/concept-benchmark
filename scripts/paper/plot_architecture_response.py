"""Build the architecture figure: accuracy under perfect interventions for every architecture.

Reads the installed results (sparse rule: results/paper/robot/grid; balanced rule: results/paper/robot/balanced_rule;
true_concepts or human_concepts, perfect interventions, k = 0, 1, 3, max) and writes a standalone TikZ panel in the
style of scripts/paper/plot_decision_support_panel.py: one line per architecture (mean over seeds, shaded mean ± SE) and a
dashed reference line, the majority-class rate (sparse rule) or the DNN's accuracy (balanced rule). `--ecbm-dir` reads the ECBM
cells from run outputs (<dir>/s<seed>/ecbm_{ideal,subconcept}.csv) instead of the installed ones. Compiles with pdflatex.

    python scripts/paper/plot_architecture_response.py --out results/paper/figures/fig_architecture_response
    python scripts/paper/plot_architecture_response.py --rule balanced --out results/paper/figures/fig_architecture_response_balanced
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from _common import BALANCED_RULE, BALANCED_TAG, SPARSE_RULE, compile_tex, mean_se
from plot_decision_support_panel import band, coords

SEEDS_BY_RULE = {
    "sparse": (1014, 1015, 1016, 1017),
    "balanced": tuple(range(1014, 1024)),
}
ARCHS = [  # (family, legend, colour, mark)
    ("cbm", "CBM", "3B6FB6", "*"),
    ("cem", "CEM", "3E8E5E", "square*"),
    ("ecbm", "ECBM", "8E5BA8", "diamond*"),
    ("probcbm", "ProbCBM", "D9822B", "triangle*"),
]
MAJORITY = 87.5  # class prior of the sparse rule

PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}\pgfplotsset{compat=1.17}\usepgfplotslibrary{fillbetween}
\usepackage{amsmath}
\renewcommand{\sfdefault}{phv}
\definecolor{annot}{HTML}{4D4D4D}
\definecolor{gridc}{HTML}{DFE4E9}
\definecolor{titlec}{HTML}{8C8C8C}
\definecolor{hair}{HTML}{B8B8B8}
%COLOURS%
\begin{document}
\begin{tikzpicture}
\pgfplotsset{
  bbvalue/.style={
    width=6.2cm, height=4.4cm, scale only axis,
    axis line style={black!55, line width=0.5pt},
    every tick/.style={black!55, line width=0.4pt},
    tick align=inside, tick pos=left,
    axis x line*=bottom, axis y line*=left,
    xtick={0,1,2,3}, xticklabels={0,1,3,max},
    xlabel style={font=\fontsize{8.5}{10}\selectfont\color{annot}, yshift=2pt},
    ylabel style={font=\fontsize{8.5}{10}\selectfont\color{annot}, yshift=-3pt},
    tick label style={font=\fontsize{8}{9}\selectfont\color{black!75}},
    yticklabel style={text width=1.05cm, align=right},
    ymajorgrids, grid style={gridc, line width=0.4pt},
    xmin=-0.35, xmax=3.35, clip=false,
    legend style={draw=none, fill=none, font=\fontsize{7.5}{9}\selectfont,
                  cells={anchor=west}, row sep=-1pt, legend image post style={scale=0.8}},
  },
  bandonly/.style={draw=none, forget plot},
}
"""


def read_cell(path: Path) -> list[float]:
    rows = sorted(
        (
            r
            for r in csv.DictReader(path.open())
            if r.get("intervention_source", "perfect") == "perfect"
        ),
        key=lambda r: int(r["budget"]),
    )
    if len(rows) != 4:
        raise ValueError(f"{path}: expected 4 budgets, got {len(rows)}")
    return [100 * float(r["accuracy"]) for r in rows]


def grid_cell(family: str, seed: int, concepts: str, rule: str = "sparse") -> Path:
    folder, tag = (
        (SPARSE_RULE, "robot__rule-sparse")
        if rule == "sparse"
        else (BALANCED_RULE, BALANCED_TAG)
    )
    matches = sorted(
        folder.glob(
            f"{tag}__concepts-{concepts}__arch-{family}__*isrc-perfect__strategy-upto__seed-{seed}__results.csv"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"{family} seed {seed}: expected one {rule}-rule file, found {[m.name for m in matches]}"
        )
    return matches[0]


def dnn_accuracies(concepts: str) -> list[float]:
    """Per-seed DNN accuracy under the balanced rule."""
    accs = []
    for seed in SEEDS_BY_RULE["balanced"]:
        path = (
            BALANCED_RULE
            / f"{BALANCED_TAG}__concepts-{concepts}__arch-cbm-and-dnn__seed-{seed}__results.csv"
        )
        accs += [
            100 * float(r["accuracy"])
            for r in csv.DictReader(path.open())
            if "dnn" in r["model"].lower()
        ]
    if len(accs) != len(SEEDS_BY_RULE["balanced"]):
        raise ValueError(f"expected one DNN accuracy per seed, got {len(accs)}")
    return accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path without extension (.tex and .pdf).",
    )
    ap.add_argument(
        "--ecbm-dir",
        type=Path,
        default=None,
        help="Read the ECBM cells from these run outputs instead.",
    )
    ap.add_argument("--concepts", choices=["true", "human"], default="true")
    ap.add_argument("--rule", choices=["sparse", "balanced"], default="sparse")
    args = ap.parse_args()
    if args.rule == "sparse":
        ref, ref_label, ref_band = MAJORITY, "majority class", ""
    else:  # DNN mean as the dashed line, mean ± SE over the same seeds as a grey band under everything
        dnn = dnn_accuracies(args.concepts)
        ref, ref_label = round(st.mean(dnn), 1), "DNN"
        se = mean_se(dnn)[1]
        ref_band = (
            rf"\addplot[draw=none, fill=annot!14, forget plot] coordinates "
            rf"{{(-0.35,{ref - se:.2f}) (3.35,{ref - se:.2f}) (3.35,{ref + se:.2f}) (-0.35,{ref + se:.2f})}} \closedcycle;"
        )

    # the CBM on human_concepts keeps the colour it has in the misspecified-concepts figure
    cbm_human = (
        "C44E52" if args.concepts == "human" and args.rule == "balanced" else None
    )
    colours = "\n".join(
        rf"\definecolor{{c{f}}}{{HTML}}{{{cbm_human if f == 'cbm' and cbm_human else hexc}}}"
        for f, _, hexc, _ in ARCHS
    )
    body, bands, all_lo, all_hi = [], [], [], []
    for family, legend, _, mark in ARCHS:
        runs = []
        for seed in SEEDS_BY_RULE[args.rule]:
            ecbm_file = (
                "ecbm_ideal.csv" if args.concepts == "true" else "ecbm_subconcept.csv"
            )
            path = (
                args.ecbm_dir / f"s{seed}/{ecbm_file}"
                if family == "ecbm" and args.ecbm_dir
                else grid_cell(family, seed, args.concepts, args.rule)
            )
            runs.append(read_cell(path))
        by_k = list(zip(*runs))
        mean = [st.mean(v) for v in by_k]
        lo, hi = zip(*[band(list(v), "se") for v in by_k])
        all_lo += lo
        all_hi += hi
        print(
            f"{legend:8} "
            + "  ".join(
                f"k={k}: {m:.1f}±{(h - m):.1f}"
                for k, m, h in zip(["0", "1", "3", "max"], mean, hi)
            )
        )
        bands.append(
            (
                family,
                rf"""\addplot[bandonly, name path={family}lo] coordinates {{{coords(lo)}}};
\addplot[bandonly, name path={family}hi] coordinates {{{coords(hi)}}};
\addplot[c{family}!20, forget plot] fill between[of={family}lo and {family}hi];""",
            )
        )
        body.append(rf"""\addplot[c{family}, line width=1.1pt, mark={mark}, mark size=1.8pt, mark options={{fill=c{family}, draw=c{family}}}] coordinates {{{coords(mean)}}};
\addlegendentry{{\textsf{{{legend}}}}}""")
    if args.rule == "sparse":  # each band right below its line
        layers = [x for (_, band), line in zip(bands, body) for x in (band, line)]
    else:  # all bands under all lines, the widest (ProbCBM) at the bottom
        order = sorted(bands, key=lambda fb: fb[0] != "probcbm")
        layers = [b for _, b in order] + body
    ymin = 5 * int(min(all_lo) // 5)
    # sparse: legend inside, top left; balanced: one row above the axis, clear of the lines
    legend_style = (
        "at={(0.02,0.90)}, anchor=north west"
        if args.rule == "sparse"
        else "at={(0.5,1.02)}, anchor=south, legend columns=4, /tikz/every even column/.append style={column sep=6pt}"
    )
    # balanced: leave room above the reference line for the concept-set label in the top-left corner
    ymax = 101 if args.rule == "sparse" else max(5 * int(max(all_hi) // 5) + 6, ref + 5)
    ticks = [t for t in range(ymin, 101, 5) if t <= ymax and abs(t - ref) > 1.5] + [ref]
    ticks = sorted(ticks)
    ytick = ",".join(f"{t:g}" for t in ticks)
    yticklabels = ",".join(f"{t:g}\\%" for t in ticks)
    tex = (
        PREAMBLE.replace("%COLOURS%", colours)
        + rf"""
\begin{{axis}}[bbvalue, at={{(0cm,0cm)}}, anchor=south west,
  xlabel={{Intervention budget $k$}}, ylabel={{Accuracy}},
  ymin={ymin}, ymax={ymax},
  ytick={{{ytick}}}, yticklabels={{{yticklabels}}},
  legend style={{{legend_style}}},
]
{ref_band}
\addplot[annot, line width=0.9pt, dashed, forget plot] coordinates {{(-0.35,{ref}) (3.35,{ref})}};
\node[anchor=south, font=\fontsize{{7}}{{8}}\selectfont, color=annot] at (axis cs:1.5,{ref}) {{{ref_label}}};
\node[anchor=north west, font=\fontsize{{8}}{{9}}\selectfont, color=annot] at (rel axis cs:0.02,0.98) {{\texttt{{{args.concepts}\_concepts}}}};
{chr(10).join(layers)}
\end{{axis}}
% same frame in every panel of this family, so that panels placed side by side line up
\pgfresetboundingbox
\path[use as bounding box] (-1.8cm,-0.92cm) rectangle (6.45cm,5.06cm);
\end{{tikzpicture}}
\end{{document}}
"""
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tex_path = args.out.with_suffix(".tex")
    tex_path.write_text(tex)
    compile_tex(tex_path)
    print(f"wrote {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
