"""Build Figure 1(a) (decision support) from balanced-rule sweep runs.

Reads either the installed files in results/paper/robot/balanced_rule (`--prefix`, the default source) or raw
`run_{tag}_s{seed}_{ground_truth,foot_subtypes}/results/` folders (robot pipeline output, `--tag`), and
writes a standalone TikZ panel in the paper's style: CBM accuracy under perfect interventions on
`true_concepts` and `human_concepts` against the DNN, shaded mean ± SE (or the range over runs), with the
Gain at k=max (CBM on true concepts minus DNN, paired per seed). Compiles the PDF with pdflatex.

    python scripts/paper/plot_decision_support_panel.py --out results/paper/figures/fig_decision_support
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

BUDGET_LABELS = ["0", "1", "3", "max"]

PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}\pgfplotsset{compat=1.17}\usepgfplotslibrary{fillbetween}
\usepackage{amsmath}
\renewcommand{\sfdefault}{phv}
\definecolor{ours}{HTML}{3B6FB6}
\definecolor{human}{HTML}{C44E52}
\definecolor{annot}{HTML}{4D4D4D}
\definecolor{gridc}{HTML}{DFE4E9}
\definecolor{titlec}{HTML}{8C8C8C}
\definecolor{hair}{HTML}{B8B8B8}
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
  bbline/.style={ours, line width=1.1pt, mark=*, mark size=1.7pt,
                 mark options={fill=ours, draw=ours}},
  bbhuman/.style={human, line width=1.1pt, mark=triangle*, mark size=2pt,
                  mark options={fill=human, draw=human}},
  bbdnn/.style={annot, line width=0.9pt, dashed},
  bandonly/.style={draw=none, forget plot},
}
"""


def read_run(run_dir: Path, variant: str) -> tuple[float, list[float]]:
    """DNN accuracy and CBM accuracy at k = 0, 1, 3, max (perfect interventions), in percent."""
    results = run_dir / "results"
    collect = next(results.glob(f"robot_{variant}_seed*_results.csv"))
    dnn = next(
        float(r["accuracy"])
        for r in csv.DictReader(collect.open())
        if r["model"] == "dnn"
    )
    cell = next(results.glob(f"robot_image_stochastic_{variant}_cbm_seed*_results.csv"))
    rows = sorted(
        (
            r
            for r in csv.DictReader(cell.open())
            if r["intervention_source"] == "perfect"
        ),
        key=lambda r: int(r["budget"]),
    )
    if len(rows) != len(BUDGET_LABELS):
        raise ValueError(
            f"{cell}: expected {len(BUDGET_LABELS)} budgets, got {len(rows)}"
        )
    return 100 * dnn, [100 * float(r["accuracy"]) for r in rows]


def read_installed(
    root: Path, prefix: str, seed: int, concepts: str
) -> tuple[float, list[float]]:
    """Same as `read_run`, from the installed results tree (results/paper/robot/balanced_rule)."""
    base = f"{prefix}__concepts-{concepts}"
    summary = root / f"{base}__arch-cbm-and-dnn__seed-{seed}__results.csv"
    dnn = next(
        float(r["accuracy"])
        for r in csv.DictReader(summary.open())
        if r["model"] == "dnn"
    )
    cell = (
        root
        / f"{base}__arch-cbm__isrc-perfect__strategy-upto__seed-{seed}__results.csv"
    )
    rows = sorted(
        (
            r
            for r in csv.DictReader(cell.open())
            if r["intervention_source"] == "perfect"
        ),
        key=lambda r: int(r["budget"]),
    )
    if len(rows) != len(BUDGET_LABELS):
        raise ValueError(
            f"{cell}: expected {len(BUDGET_LABELS)} budgets, got {len(rows)}"
        )
    return 100 * dnn, [100 * float(r["accuracy"]) for r in rows]


def band(values: list[float], kind: str) -> tuple[float, float]:
    if kind == "range":
        return min(values), max(values)
    mean, se = mean_se(values)
    return mean - se, mean + se


def coords(values) -> str:
    return " ".join(f"({i},{v:.2f})" for i, v in enumerate(values))


def build_tex(true_runs, human_runs, dnn_runs, gains, band_kind: str) -> str:
    true_k = list(zip(*true_runs))
    human_k = list(zip(*human_runs))
    true_mean = [st.mean(v) for v in true_k]
    human_mean = [st.mean(v) for v in human_k]
    dnn_mean = st.mean(dnn_runs)
    dnn_lo, dnn_hi = band(dnn_runs, band_kind)
    gain, gain_se = mean_se(gains)
    true_lo, true_hi = zip(*[band(list(v), band_kind) for v in true_k])
    human_lo, human_hi = zip(*[band(list(v), band_kind) for v in human_k])
    top = max(true_hi[3], true_mean[3])
    ymin = 5 * int(min(human_lo) // 5) - 1
    ymax = top + 4.0
    return (
        PREAMBLE
        + rf"""
\begin{{axis}}[bbvalue, at={{(0cm,0cm)}}, anchor=south west,
  xlabel={{Intervention budget $k$}}, ylabel={{Accuracy}},
  ymin={ymin:.0f}, ymax={ymax:.1f},
  ytick={{80,85,{dnn_mean:.1f},{true_mean[3]:.1f}}}, yticklabels={{80\%,85\%,{dnn_mean:.1f}\%,{true_mean[3]:.1f}\%}},
  legend pos=south east,
]
\addplot[bandonly, name path=dnnlo] coordinates {{(-0.35,{dnn_lo:.2f}) (3.35,{dnn_lo:.2f})}};
\addplot[bandonly, name path=dnnhi] coordinates {{(-0.35,{dnn_hi:.2f}) (3.35,{dnn_hi:.2f})}};
\addplot[annot!25, forget plot] fill between[of=dnnlo and dnnhi];
\addplot[bandonly, name path=alo] coordinates {{{coords(true_lo)}}};
\addplot[bandonly, name path=ahi] coordinates {{{coords(true_hi)}}};
\addplot[ours!22, forget plot] fill between[of=alo and ahi];
\addplot[bandonly, name path=hlo] coordinates {{{coords(human_lo)}}};
\addplot[bandonly, name path=hhi] coordinates {{{coords(human_hi)}}};
\addplot[human!20, forget plot] fill between[of=hlo and hhi];
\addplot[bbline] coordinates {{{coords(true_mean)}}};
\addlegendentry{{\textsf{{CBM}}, \texttt{{true\_concepts}}}}
\addplot[bbhuman] coordinates {{{coords(human_mean)}}};
\addlegendentry{{\textsf{{CBM}}, \texttt{{human\_concepts}}}}
\addplot[bbdnn] coordinates {{(-0.35,{dnn_mean:.2f}) (3.35,{dnn_mean:.2f})}};
\addlegendentry{{\textsf{{DNN}}}}
\draw[annot, line width=0.6pt, <->, >=latex]
  (axis cs:3.13,{dnn_mean:.2f}) -- (axis cs:3.13,{true_mean[3]:.2f});
\node[anchor=east, font=\fontsize{{8}}{{9}}\selectfont, color=black]
  at (axis cs:3.35,{top + 1.3:.2f}) {{\textsf{{Gain}}\,$={gain:.1f}\pm{gain_se:.1f}\%$}};
\end{{axis}}
% same frame in every panel of this family, so that panels placed side by side line up
\pgfresetboundingbox
\path[use as bounding box] (-1.8cm,-0.92cm) rectangle (6.45cm,5.06cm);
\end{{tikzpicture}}
\end{{document}}
"""
    )


def parse_seeds(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return (
        list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in text.split(",")]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Folder of raw run_{tag}_s{seed}_{preset} runs, with --tag (default: the results root's balanced_rule folder).",
    )
    ap.add_argument(
        "--prefix", default=BALANCED_TAG, help="File-name prefix of the installed runs."
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Read raw run folders with this tag instead.",
    )
    ap.add_argument(
        "--seeds", default="1014-1023", help="Seed range 'a-b' or list 'a,b,c'."
    )
    ap.add_argument(
        "--band",
        choices=["se", "range"],
        default="se",
        help="Shading: mean ± SE (default) or range over runs.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path without extension (.tex and .pdf).",
    )
    add_results_root(ap)
    args = ap.parse_args()
    use_results_root(args)
    args.root = args.root or PAPER.balanced_rule

    seeds = parse_seeds(args.seeds)
    true_runs, human_runs, dnn_runs, gains = [], [], [], []
    for seed in seeds:
        if args.tag:
            dnn, true_acc = read_run(
                args.root / f"run_{args.tag}_s{seed}_ground_truth", "ideal"
            )
            _, human_acc = read_run(
                args.root / f"run_{args.tag}_s{seed}_foot_subtypes", "subconcept"
            )
        else:
            dnn, true_acc = read_installed(args.root, args.prefix, seed, "true")
            _, human_acc = read_installed(args.root, args.prefix, seed, "human")
        true_runs.append(true_acc)
        human_runs.append(human_acc)
        dnn_runs.append(dnn)
        gains.append(true_acc[3] - dnn)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tex_path = args.out.with_suffix(".tex")
    tex_path.write_text(build_tex(true_runs, human_runs, dnn_runs, gains, args.band))
    compile_tex(tex_path)
    print(
        f"wrote {tex_path.with_suffix('.pdf')}  (n={len(seeds)}, Gain {st.mean(gains):.1f}, band {args.band})"
    )


if __name__ == "__main__":
    main()
