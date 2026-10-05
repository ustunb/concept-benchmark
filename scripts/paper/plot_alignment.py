"""Alignment figure under the balanced rule: what constraining the HasKnees weight does to the CBM.

Reads the per-seed results of scripts/paper/evaluate_alignment_balanced.py and draws two panels of horizontal bars, one pair per
concept set (unconstrained vs constrained CBM; mean over seeds, error bars = SE):
  left   accuracy before interventions (k = 0);
  right  accuracy after interventions (k = max).
The benefit of interventions (mean accuracy over k in {1, 3, max} minus accuracy at k = 0) is printed for the text.

    PYTHONPATH=. python scripts/paper/plot_alignment.py --out results/paper/figures/fig_alignment
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from _common import (
    BALANCED_TAG,
    PAPER,
    add_results_root,
    compile_tex,
    mean_se,
    use_results_root,
)

RESULTS_NAME = f"{BALANCED_TAG}__arch-cbm__isrc-perfect__alignment.csv"
SETS = (("true", 0), ("human", 1))  # concept set, row (bottom to top)


def per_seed() -> dict[tuple[str, str, str], list[float]]:
    """(metric, concept set, before|after) -> per-seed values in percent / points."""
    out: dict = {}
    for r in csv.DictReader((PAPER.alignment / RESULTS_NAME).open()):
        for which in ("before", "after"):
            k0 = float(r[f"k0_{which}"])
            after = st.mean(float(r[f"k{k}_{which}"]) for k in ("1", "3", "max"))
            out.setdefault(("k0", r["concepts"], which), []).append(k0)
            out.setdefault(("delta", r["concepts"], which), []).append(after - k0)
            out.setdefault(("kmax", r["concepts"], which), []).append(
                float(r[f"kmax_{which}"])
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path without extension (.tex and .pdf).",
    )
    add_results_root(ap)
    args = ap.parse_args()
    use_results_root(args)
    values = per_seed()
    stat = {k: mean_se(v) for k, v in values.items()}
    for (metric, c, which), (m, se) in sorted(stat.items()):
        print(
            f"{metric:6s} {c:6s} {which:6s} {m:6.2f} ± {se:.2f}  (n={len(values[(metric, c, which)])})"
        )

    def bars(metric: str, which: str) -> str:
        return " ".join(
            f"({stat[(metric, c, which)][0]:.2f},{y}) +- ({stat[(metric, c, which)][1]:.2f},0)"
            for c, y in SETS
        )

    def labels(metric: str, which: str, shift: str, signed: bool) -> str:
        rows = []
        for c, y in SETS:
            m, se = stat[(metric, c, which)]
            text = f"{m:+.1f}" if signed else f"{m:.1f}"
            rows.append(
                rf"\node[anchor=west, font=\fontsize{{7.5}}{{9}}\selectfont, yshift={shift}] at (axis cs:{m + se:.2f},{y}) {{${text}\%$}};"
            )
        return "\n".join(rows)

    def panel(
        metric: str,
        at: str,
        xlabel: str,
        extra: str,
        signed: bool,
        ticklabels: bool,
        legend: bool,
    ) -> str:
        yt = (
            r"yticklabels={\texttt{true\_concepts},\texttt{human\_concepts}}"
            if ticklabels
            else "yticklabels={,}"
        )
        leg = (
            r"""legend to name=alignlegend, legend style={draw=none, fill=none, font=\fontsize{7.5}{9}\selectfont, legend columns=2,
                /tikz/every even column/.append style={column sep=8pt}},
  legend image code/.code={\draw[#1, draw=none] (0cm,-0.09cm) rectangle (0.32cm,0.09cm);},"""
            if legend
            else ""
        )
        entries = (
            (
                r"\addlegendentry{\textsf{CBM}}",
                r"\addlegendentry{Constrained \textsf{CBM}}",
            )
            if legend
            else ("", "")
        )
        return rf"""\begin{{axis}}[bars, at={{({at},0cm)}}, anchor=south west, {yt}, xlabel={{{xlabel}}}, {extra}, {leg}]
\addplot[fill=cons, draw=none] coordinates {{{bars(metric, "after")}}};
{entries[1]}
\addplot[fill=uncons, draw=none] coordinates {{{bars(metric, "before")}}};
{entries[0]}
{labels(metric, "after", "-5.2pt", signed)}
{labels(metric, "before", "5.2pt", signed)}
\end{{axis}}"""

    tex = (
        r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\definecolor{cons}{HTML}{E3A9AB}
\definecolor{uncons}{HTML}{C44E52}
\definecolor{annot}{HTML}{4D4D4D}
\definecolor{gridc}{HTML}{DFE4E9}
\begin{document}
\begin{tikzpicture}
\pgfplotsset{
  bars/.style={
    xbar, bar width=11pt, width=3.6cm, height=3.0cm, scale only axis, clip=false,
    axis line style={black!55, line width=0.5pt}, every tick/.style={black!55, line width=0.4pt},
    axis x line*=bottom, axis y line*=left, tick align=outside, ytick style={draw=none},
    ytick={0,1}, ymin=-0.6, ymax=1.6, yticklabel style={font=\fontsize{8}{9}\selectfont},
    xticklabel={$\pgfmathprintnumber{\tick}\%$}, xticklabel style={font=\fontsize{8}{9}\selectfont},
    xmajorgrids, grid style={gridc, line width=0.4pt},
    xlabel style={font=\fontsize{8.5}{10}\selectfont\color{annot}},
    error bars/x dir=both, error bars/x explicit, error bars/error bar style={annot, line width=0.6pt},
  },
}
"""
        + panel(
            "k0",
            "0cm",
            "Accuracy before interventions",
            "xmin=70, xmax=96, xtick={70,80,90}",
            False,
            True,
            True,
        )
        + "\n"
        + panel(
            "kmax",
            "4.5cm",
            "Accuracy after interventions",
            "xmin=70, xmax=96, xtick={70,80,90}",
            False,
            False,
            False,
        )
        + r"""
\node[anchor=south] at (4.05cm,3.15cm) {\pgfplotslegendfromname{alignlegend}};
\end{tikzpicture}
\end{document}
"""
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tex_path = args.out.with_suffix(".tex")
    tex_path.write_text(tex)
    compile_tex(
        tex_path, passes=2
    )  # the second pass places the legend drawn outside the axes
    print(f"wrote {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
