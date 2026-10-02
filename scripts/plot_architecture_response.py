"""Build the section 2 figure: accuracy under perfect interventions for every architecture (sparse rule).

Reads the installed grid (results/paper/robot/grid, true_concepts or human_concepts, perfect interventions, k = 0, 1, 3, max) and
writes a standalone TikZ panel in the style of scripts/plot_decision_support_panel.py: one line per architecture
(mean over seeds, shaded mean ± SE) and the majority-class rate as a dashed line. Until the retrained ECBM cells
are installed, `--ecbm-dir` reads them from the run outputs (<dir>/s<seed>/ecbm_{ideal,subconcept}.csv). Compiles with pdflatex.

    python scripts/plot_architecture_response.py --out results/paper/figures/fig_architecture_response
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import subprocess
from pathlib import Path

from plot_decision_support_panel import band, coords

REPO = Path(__file__).resolve().parent.parent
GRID = REPO / "results/paper/robot/grid"
SEEDS = (1014, 1015, 1016, 1017)
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
        (r for r in csv.DictReader(path.open()) if r.get("intervention_source", "perfect") == "perfect"),
        key=lambda r: int(r["budget"]),
    )
    if len(rows) != 4:
        raise ValueError(f"{path}: expected 4 budgets, got {len(rows)}")
    return [100 * float(r["accuracy"]) for r in rows]


def grid_cell(family: str, seed: int, concepts: str) -> Path:
    matches = sorted(GRID.glob(f"robot__rule-sparse__concepts-{concepts}__arch-{family}__*isrc-perfect__strategy-upto__seed-{seed}__results.csv"))
    if len(matches) != 1:
        raise ValueError(f"{family} seed {seed}: expected one grid file, found {[m.name for m in matches]}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Output path without extension (.tex and .pdf).")
    ap.add_argument("--ecbm-dir", type=Path, default=None, help="Retrained ECBM run outputs (until installed).")
    ap.add_argument("--concepts", choices=["true", "human"], default="true")
    args = ap.parse_args()

    colours = "\n".join(rf"\definecolor{{c{f}}}{{HTML}}{{{hexc}}}" for f, _, hexc, _ in ARCHS)
    body, all_lo = [], []
    for family, legend, _, mark in ARCHS:
        runs = []
        for seed in SEEDS:
            ecbm_file = "ecbm_ideal.csv" if args.concepts == "true" else "ecbm_subconcept.csv"
            path = args.ecbm_dir / f"s{seed}/{ecbm_file}" if family == "ecbm" and args.ecbm_dir else grid_cell(family, seed, args.concepts)
            runs.append(read_cell(path))
        by_k = list(zip(*runs))
        mean = [st.mean(v) for v in by_k]
        lo, hi = zip(*[band(list(v), "se") for v in by_k])
        all_lo += lo
        print(f"{legend:8} " + "  ".join(f"k={k}: {m:.1f}±{(h - m):.1f}" for k, m, h in zip(["0", "1", "3", "max"], mean, hi)))
        body.append(rf"""\addplot[bandonly, name path={family}lo] coordinates {{{coords(lo)}}};
\addplot[bandonly, name path={family}hi] coordinates {{{coords(hi)}}};
\addplot[c{family}!20, forget plot] fill between[of={family}lo and {family}hi];""")
        body.append(rf"""\addplot[c{family}, line width=1.1pt, mark={mark}, mark size=1.8pt, mark options={{fill=c{family}, draw=c{family}}}] coordinates {{{coords(mean)}}};
\addlegendentry{{\textsf{{{legend}}}}}""")
    ymin = 5 * int(min(all_lo) // 5)
    tex = PREAMBLE.replace("%COLOURS%", colours) + rf"""
\begin{{axis}}[bbvalue, at={{(0cm,0cm)}}, anchor=south west,
  xlabel={{Intervention budget $k$}}, ylabel={{Accuracy}},
  ymin={ymin}, ymax=101,
  ytick={{80,85,{MAJORITY},90,95,100}}, yticklabels={{80\%,85\%,{MAJORITY}\%,90\%,95\%,100\%}},
  legend pos=south east,
]
\addplot[annot, line width=0.9pt, dashed, forget plot] coordinates {{(-0.35,{MAJORITY}) (3.35,{MAJORITY})}};
\node[anchor=south west, font=\fontsize{{7}}{{8}}\selectfont, color=annot] at (axis cs:-0.3,{MAJORITY}) {{majority class}};
\node[anchor=north west, font=\fontsize{{8}}{{9}}\selectfont, color=annot] at (rel axis cs:0.02,0.98) {{\texttt{{{args.concepts}\_concepts}}}};
{chr(10).join(body)}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tex_path = args.out.with_suffix(".tex")
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                   cwd=tex_path.parent, check=True, stdout=subprocess.DEVNULL)
    for ext in (".aux", ".log"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    print(f"wrote {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
