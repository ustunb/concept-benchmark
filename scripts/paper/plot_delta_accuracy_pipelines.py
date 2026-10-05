"""Build the concept-source figure: change in accuracy from interventions across concept and intervention sources.

Reads the installed results: sparse rule (results/paper/robot/grid, default) or balanced rule
(results/paper/robot/balanced_rule, `--rule balanced`; interventions on the label-free CBM set a concept to its
5th/95th-percentile score). For each architecture, concept source and
intervention source, Delta Accuracy = mean accuracy over k in {1, 3, max} minus accuracy at k = 0, computed per seed
and shown as mean +/- SE over seeds (LLM interventions: Gemini 2.5 Flash-Lite at 224 px, seeds 1015-1017).
Layouts (same pgfplots style and colours as scripts/paper/plot_architecture_response.py):
  A  x = seven pipelines (human x {perfect, expert, llm}; machine x {expert, llm}; llm x llm; clip x llm)
  B  A plus a true_concepts + perfect reference column
  C  three panels (perfect | expert | llm interventions), x = concept source, lines with SE bands
  D  C as points with SE error bars, no connecting lines
  E  heatmap, rows = architecture, columns = all 15 pipelines, Delta Accuracy printed in each cell
`--yaxis zoom` (A-D) clips the axis around 0 and marks off-scale values with an arrow and their value.

    python scripts/paper/plot_delta_accuracy_pipelines.py --layout A --yaxis linear --out results/paper/figures/fig_delta_accuracy
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from _common import BALANCED_RULE, BALANCED_TAG, SPARSE_RULE, compile_tex, mean_se, read_budget_rows
from plot_architecture_response import ARCHS

SOURCE = {"sparse": (SPARSE_RULE, "robot__rule-sparse"), "balanced": (BALANCED_RULE, BALANCED_TAG)}
RULE = "sparse"  # set from --rule in main()
SEEDS_BY_RULE = {"sparse": (1014, 1015, 1016, 1017), "balanced": tuple(range(1014, 1024))}
CONCEPTS = [("true", r"\texttt{true\_concepts}"), ("human", r"\texttt{human\_concepts}"),
            ("machine", r"\texttt{machine\_annotation}"), ("llm", r"\texttt{llm\_concepts}"),
            ("clip", r"\texttt{clip\_concepts}")]
ISRC = [("perfect", "isrc-perfect", "perfect"), ("expert", "isrc-expert", "expert"), ("llm", "isrc-llm", "LLM")]
PIPELINES_A = [("human", "perfect"), ("human", "expert"), ("human", "llm"), ("machine", "expert"),
               ("machine", "llm"), ("llm", "llm"), ("clip", "llm")]
ZOOM = (-12, 14)

PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}\pgfplotsset{compat=1.17}\usepgfplotslibrary{fillbetween,groupplots}
\usepackage{amsmath}
\renewcommand{\sfdefault}{phv}
\definecolor{annot}{HTML}{4D4D4D}
\definecolor{gridc}{HTML}{DFE4E9}
%COLOURS%
\begin{document}
\begin{tikzpicture}
\pgfplotsset{
  bbvalue/.style={
    scale only axis,
    axis line style={black!55, line width=0.5pt},
    every tick/.style={black!55, line width=0.4pt},
    tick align=inside, tick pos=left,
    axis x line*=bottom, axis y line*=left,
    xlabel style={font=\fontsize{8.5}{10}\selectfont\color{annot}},
    ylabel style={font=\fontsize{8.5}{10}\selectfont\color{annot}, yshift=-3pt},
    tick label style={font=\fontsize{7.5}{9}\selectfont\color{black!75}},
    xticklabel style={align=center},
    ymajorgrids, grid style={gridc, line width=0.4pt},
    clip=false,
    legend style={draw=none, fill=none, font=\fontsize{7.5}{9}\selectfont, legend columns=4,
                  /tikz/every even column/.append style={column sep=6pt}},
  },
  bandonly/.style={draw=none, forget plot},
}
"""


def delta(family: str, concepts: str, isrc_tag: str) -> list[float]:
    """Per-seed Delta Accuracy (points) for one cell."""
    out = []
    folder, tag = SOURCE[RULE]
    for f in sorted(folder.glob(f"{tag}__concepts-{concepts}__arch-{family}__*{isrc_tag}*__results.csv")):
        if isrc_tag == "isrc-llm" and "img-224px" not in f.name:
            continue
        # under the balanced rule, the label-free CBM's interventions use the 5th/95th percentile of its training scores
        if family == "cbm" and RULE == "balanced" and concepts in ("machine", "llm", "clip"):
            if "enc-koh595" not in f.name:
                continue
        elif "enc-koh595" in f.name:
            continue
        if int(f.name.split("__seed-")[1].split("__")[0]) not in SEEDS_BY_RULE[RULE]:
            continue
        acc = [100 * float(r["accuracy"]) for r in read_budget_rows(f)]
        out.append(st.mean(acc[1:]) - acc[0])
    return out


def stats(values: list[float]) -> tuple[float, float]:
    return mean_se(values) if len(values) > 1 else (st.mean(values), 0.0)


def isrc_tag(name: str) -> str:
    return next(t for n, t, _ in ISRC if n == name)


def tick(concepts: str, isrc: str) -> str:
    label = next(lbl for n, lbl in CONCEPTS if n == concepts)
    return label + r"\\" + next(d for n, _, d in ISRC if n == isrc)


def clipped(mean: float, lo: float, hi: float, zoom: bool):
    """(plotted value, band low, band high, off-scale marker or None)."""
    if not zoom:
        return mean, lo, hi, None
    a, b = ZOOM
    if mean < a or mean > b:
        return None, None, None, mean
    return mean, max(lo, a), min(hi, b), None


def lines_body(pipelines, zoom: bool, x0: float = 0.0) -> tuple[str, list[str]]:
    body, offscale = [], []
    for family, legend, _, mark in ARCHS:
        pts, los, his = [], [], []
        for i, (c, s) in enumerate(pipelines):
            m, se = stats(delta(family, c, isrc_tag(s)))
            v, lo, hi, off = clipped(m, m - se, m + se, zoom)
            if off is not None:
                y = ZOOM[0] if off < ZOOM[0] else ZOOM[1]
                offscale.append(rf"\draw[c{family}, -latex, line width=0.8pt] (axis cs:{x0 + i:.2f},{y + (3 if y < 0 else -3)}) -- (axis cs:{x0 + i:.2f},{y});"
                                rf"\node[c{family}, font=\fontsize{{6.5}}{{7}}\selectfont, anchor={'south' if y < 0 else 'north'}] at (axis cs:{x0 + i:.2f},{y + (3 if y < 0 else -3)}) {{${off:+.0f}$}};")
                continue
            pts.append((x0 + i, v)); los.append((x0 + i, lo)); his.append((x0 + i, hi))
        c = lambda xs: " ".join(f"({x:.2f},{y:.2f})" for x, y in xs)
        if pts:
            body.append(rf"\addplot[bandonly, name path={family}lo] coordinates {{{c(los)}}};"
                        rf"\addplot[bandonly, name path={family}hi] coordinates {{{c(his)}}};"
                        rf"\addplot[c{family}!20, forget plot] fill between[of={family}lo and {family}hi];")
        body.append(rf"\addplot[c{family}, line width=1.1pt, mark={mark}, mark size=1.8pt, mark options={{fill=c{family}, draw=c{family}}}] coordinates {{{c(pts)}}};"
                    rf"\addlegendentry{{\textsf{{{legend}}}}}")
    return "\n".join(body), offscale


def points_body(concepts, isrc, zoom: bool) -> tuple[str, list[str]]:
    body, offscale = [], []
    offsets = {f: (k - 1.5) * 0.12 for k, (f, *_ ) in enumerate(ARCHS)}
    for family, legend, _, mark in ARCHS:
        rows = []
        for i, c in enumerate(concepts):
            vals = delta(family, c, isrc_tag(isrc))
            if not vals:
                continue
            m, se = stats(vals)
            v, lo, hi, off = clipped(m, m - se, m + se, zoom)
            x = i + offsets[family]
            if off is not None:
                y = ZOOM[0] if off < ZOOM[0] else ZOOM[1]
                offscale.append(rf"\draw[c{family}, -latex, line width=0.8pt] (axis cs:{x:.2f},{y + 3}) -- (axis cs:{x:.2f},{y});"
                                rf"\node[c{family}, font=\fontsize{{6}}{{7}}\selectfont, anchor=south] at (axis cs:{x:.2f},{y + 3}) {{${off:+.0f}$}};")
                continue
            rows.append(f"({x:.2f},{v:.2f}) +- (0,{min(se, hi - v):.2f})")
        body.append(rf"\addplot[c{family}, only marks, mark={mark}, mark size=1.8pt, mark options={{fill=c{family}, draw=c{family}}}, error bars/.cd, y dir=both, y explicit] coordinates {{{' '.join(rows)}}};"
                    rf"\addlegendentry{{\textsf{{{legend}}}}}")
    return "\n".join(body), offscale


def build(layout: str, yaxis: str) -> str:
    zoom = yaxis == "zoom"
    colours = "\n".join(rf"\definecolor{{c{f}}}{{HTML}}{{{h}}}" for f, _, h, _ in ARCHS)
    yrange = f"ymin={ZOOM[0]}, ymax={ZOOM[1]}," if zoom else ("ymin=-60, ymax=20," if layout in "CD" else "")
    zero = r"\addplot[annot, dashed, line width=0.8pt, forget plot] coordinates {(-0.5,0) (%s,0)};"
    if layout in "AB":
        pipelines = ([("true", "perfect")] if layout == "B" else []) + PIPELINES_A
        body, off = lines_body(pipelines, zoom)
        ticks = ",".join(tick(c, s) for c, s in pipelines)
        n = len(pipelines)
        tex = rf"""
\begin{{axis}}[bbvalue, width={1.45 * n:.1f}cm, height=4.4cm, {yrange}
  xtick={{{",".join(str(i) for i in range(n))}}}, xticklabels={{{ticks}}}, xmin=-0.5, xmax={n - 0.5},
  ylabel={{$\Delta$ Accuracy (points)}},
  legend style={{at={{(0.5,1.03)}}, anchor=south}}]
{zero % (n - 0.5)}
{body}
{chr(10).join(off)}
\end{{axis}}"""
    elif layout in "CD":
        names = [c for c, _ in CONCEPTS]
        panels = []
        for k, (isrc, _, title) in enumerate(ISRC):
            concepts = [c for c in names if delta("cbm", c, isrc_tag(isrc))]
            if layout == "C":
                body, off = lines_body([(c, isrc) for c in concepts], zoom)
            else:
                body, off = points_body(concepts, isrc, zoom)
            labels = ",".join(next(l for n, l in CONCEPTS if n == c) for c in concepts)
            legend = "" if k == 1 else r"legend to name=dummy,"
            ylabel = r", ylabel={$\Delta$ Accuracy (points)}" if k == 0 else ""
            panels.append(rf"""\nextgroupplot[title={{{title} interventions}}, {legend} xtick={{{",".join(str(i) for i in range(len(concepts)))}}},
  xticklabels={{{labels}}}, xmin=-0.5, xmax={len(concepts) - 0.5}{ylabel}]
{zero % (len(concepts) - 0.5)}
{body}
{chr(10).join(off)}""")
        tex = rf"""
\begin{{groupplot}}[group style={{group size=3 by 1, horizontal sep=0.9cm, y descriptions at=edge left}},
  bbvalue, width=4.8cm, height=4.2cm, {yrange} title style={{font=\fontsize{{8}}{{9}}\selectfont\color{{annot}}}},
  legend style={{at={{(0.5,1.15)}}, anchor=south}}]
{chr(10).join(panels)}
\end{{groupplot}}"""
    else:  # E: heatmap
        cols = [(c, s) for c, _ in CONCEPTS for s, _, _ in ISRC if delta("cbm", c, isrc_tag(s))]  # grouped by concept source
        cells = []
        for r, (family, legend, _, _) in enumerate(ARCHS):
            for j, (c, s) in enumerate(cols):
                vals = delta(family, c, isrc_tag(s))
                m, se = stats(vals) if vals else (float("nan"), float("nan"))
                cells.append((j, len(ARCHS) - 1 - r, m, se))
        coords = " ".join(f"({j},{r}) [{m:.2f}]" for j, r, m, _ in cells)
        text = "\n".join(rf"\node[font=\fontsize{{6}}{{7}}\selectfont, align=center, color={'white' if abs(m) > 20 else 'black'}] at (axis cs:{j},{r}) {{{m:.1f}\%\\[-1pt]{{\fontsize{{4.5}}{{5}}\selectfont $\pm${se:.1f}}}}};"
                         for j, r, m, se in cells)
        # one intervention source per column; each concept source named once, centred under its columns
        ticks = ",".join(rf"\texttt{{{s}}}" for _, s in cols)  # the robot table's names: perfect, expert, llm
        groups = {}
        for j, (c, _) in enumerate(cols):
            groups.setdefault(c, []).append(j)
        text += "\n" + "\n".join(
            rf"\node[font=\fontsize{{6.5}}{{7}}\selectfont, anchor=north] at (axis cs:{sum(js) / len(js)},-1.25) "
            rf"{{{next(lbl for n, lbl in CONCEPTS if n == c)}}};" for c, js in groups.items())
        text += ("\n" + rf"\node[font=\fontsize{{6.5}}{{7}}\selectfont\bfseries, anchor=east] at (axis cs:-0.55,-0.8) {{Intervention}};"
                 + "\n" + rf"\node[font=\fontsize{{6.5}}{{7}}\selectfont\bfseries, anchor=north east] at (axis cs:-0.55,-1.25) {{Dataset}};")
        arch_labels = ",".join(rf"\textsf{{{l}}}" for _, l, _, _ in reversed(ARCHS))
        tex = rf"""
\begin{{axis}}[bbvalue, width={0.95 * len(cols):.1f}cm, height=3.2cm, axis line style={{draw=none}}, ymajorgrids=false,
  colormap={{rdbu}}{{rgb255=(178,24,43) rgb255=(244,165,130) rgb255=(247,247,247) rgb255=(146,197,222) rgb255=(33,102,172)}},
  point meta min=-30, point meta max=30, colorbar, colorbar style={{width=0.25cm, ylabel={{$\Delta$ Accuracy}}, ytick={{-30,0,30}},
  yticklabel style={{font=\fontsize{{6.5}}{{7}}\selectfont}}}},
  xtick={{{",".join(str(i) for i in range(len(cols)))}}}, xticklabels={{{ticks}}}, xticklabel style={{font=\fontsize{{6.5}}{{7}}\selectfont, align=center}},
  ytick={{0,1,2,3}}, yticklabels={{{arch_labels}}},
  enlargelimits=false, clip=false, xmin=-0.5, xmax={len(cols) - 0.5}, ymin=-0.5, ymax=3.5]
\addplot[matrix plot*, mesh/cols={len(cols)}, point meta=explicit] coordinates {{{coords}}};
{text}
\end{{axis}}"""
    return PREAMBLE.replace("%COLOURS%", colours) + tex + "\n\\end{tikzpicture}\n\\end{document}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=list("ABCDE"), required=True)
    ap.add_argument("--yaxis", choices=["linear", "zoom"], default="linear")
    ap.add_argument("--rule", choices=["sparse", "balanced"], default="sparse")
    ap.add_argument("--out", type=Path, required=True, help="Output path without extension (.tex and .pdf).")
    args = ap.parse_args()
    global RULE
    RULE = args.rule
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tex_path = args.out.with_suffix(".tex")
    tex_path.write_text(build(args.layout, args.yaxis))
    compile_tex(tex_path)
    print(f"wrote {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
