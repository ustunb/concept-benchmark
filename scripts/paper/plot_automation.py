"""Automation figure: work a CBM automates on sudoku against a DNN, as the intervention budget grows.

Reads the installed sudoku cells (one resolution, selective-accuracy target tau = 0.95) and draws the CBM's coverage and
net work automated at k = 0, 1, 3, max (mean over seeds, shaded mean ± SE) with the DNN's coverage as a dashed line.
Same style as scripts/paper/plot_architecture_response.py; compiles with pdflatex.

    python scripts/paper/plot_automation.py --out results/paper/figures/fig_automation
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "paper"), str(REPO / "scripts")]
from make_sudoku_table import SELECTIVE, TAU, read_cells  # noqa: E402


def mean_se(values: list[float]) -> tuple[float, float]:
    return st.mean(values), st.stdev(values) / len(values) ** 0.5


def dnn_coverage(res: int) -> list[float]:
    values = []
    for f in sorted(SELECTIVE.glob(f"sudoku__*__res-{res}px__threshold-per-budget__seed-*__selective-all-tau.csv")):
        values += [100 * float(r["selective_cov"]) for r in csv.DictReader(f.open())
                   if r["model"] == "dnn" and r["selective_cov"] and abs(float(r["target_accuracy"]) - float(TAU)) < 1e-9]
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=18)
    ap.add_argument("--family", default="cbm")
    ap.add_argument("--out", type=Path, required=True, help="Output path without extension (.tex and .pdf).")
    args = ap.parse_args()
    net, coverage = read_cells(args.res, args.family)
    dnn = dnn_coverage(args.res)
    if len(dnn) != len(net):
        raise SystemExit(f"{len(net)} seeds for the model but {len(dnn)} for the DNN")
    series = {name: [mean_se([r[k] for r in runs]) for k in range(4)] for name, runs in (("net", net), ("cov", coverage))}
    d_mean, d_se = mean_se(dnn)
    print(f"{args.family} at {args.res}px, n={len(net)}: coverage " + " ".join(f"{m:.1f}±{s:.1f}" for m, s in series["cov"])
          + " | net " + " ".join(f"{m:.1f}±{s:.1f}" for m, s in series["net"]) + f" | DNN {d_mean:.1f}±{d_se:.1f}")

    def line(key: str) -> str:
        return " ".join(f"({k},{m:.2f})" for k, (m, _) in enumerate(series[key]))

    def edge(key: str, sign: int) -> str:
        return " ".join(f"({k},{min(100, max(0, m + sign * s)):.2f})" for k, (m, s) in enumerate(series[key]))

    tex = rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{fillbetween}}
\pgfplotsset{{compat=1.17}}
\definecolor{{ccov}}{{HTML}}{{3E8E5E}}
\definecolor{{cnet}}{{HTML}}{{3B6FB6}}
\definecolor{{annot}}{{HTML}}{{4D4D4D}}
\definecolor{{gridc}}{{HTML}}{{DFE4E9}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=6.2cm, height=4.4cm, scale only axis,
  axis line style={{black!55, line width=0.5pt}}, every tick/.style={{black!55, line width=0.4pt}},
  axis x line*=bottom, axis y line*=left, tick align=outside,
  xtick={{0,1,2,3}}, xticklabels={{0,1,3,max}}, xmin=-0.35, xmax=3.35,
  ymin=0, ymax=105, ytick={{0,25,50,75,100}}, yticklabel={{$\pgfmathprintnumber{{\tick}}\%$}},
  ticklabel style={{font=\fontsize{{8}}{{9}}\selectfont}},
  ymajorgrids, grid style={{gridc, line width=0.4pt}},
  xlabel={{Intervention budget $k$}}, ylabel={{Boards}},
  xlabel style={{font=\fontsize{{8.5}}{{10}}\selectfont\color{{annot}}, yshift=2pt}},
  ylabel style={{font=\fontsize{{8.5}}{{10}}\selectfont\color{{annot}}, yshift=-3pt}},
  legend style={{draw=none, fill=none, font=\fontsize{{7.5}}{{9}}\selectfont, cells={{anchor=west}}, at={{(0.03,0.40)}}, anchor=west,
                row sep=-1pt, legend image post style={{scale=0.8}}}},
]
\addplot[draw=none, fill=annot!14, forget plot] coordinates {{(-0.35,{d_mean - d_se:.2f}) (3.35,{d_mean - d_se:.2f}) (3.35,{d_mean + d_se:.2f}) (-0.35,{d_mean + d_se:.2f})}} \closedcycle;
\addplot[draw=none, forget plot, name path=clo] coordinates {{{edge("cov", -1)}}};
\addplot[draw=none, forget plot, name path=chi] coordinates {{{edge("cov", 1)}}};
\addplot[ccov!20, forget plot] fill between[of=clo and chi];
\addplot[draw=none, forget plot, name path=nlo] coordinates {{{edge("net", -1)}}};
\addplot[draw=none, forget plot, name path=nhi] coordinates {{{edge("net", 1)}}};
\addplot[cnet!20, forget plot] fill between[of=nlo and nhi];
\addplot[ccov, line width=1.1pt, mark=square*, mark size=1.8pt, mark options={{fill=ccov, draw=ccov}}] coordinates {{{line("cov")}}};
\addlegendentry{{\textsf{{CBM}} \textsf{{Coverage}}}}
\addplot[cnet, line width=1.1pt, mark=*, mark size=1.8pt, mark options={{fill=cnet, draw=cnet}}] coordinates {{{line("net")}}};
\addlegendentry{{\textsf{{CBM}} \textsf{{NetWorkAutomated}}}}
\addplot[annot, line width=0.9pt, dashed] coordinates {{(-0.35,{d_mean:.2f}) (3.35,{d_mean:.2f})}};
\addlegendentry{{\textsf{{DNN}}}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tex_path = args.out.with_suffix(".tex")
    tex_path.write_text(tex)
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                            cwd=tex_path.parent, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stdout[-2500:])
    for ext in (".aux", ".log"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    print(f"wrote {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
