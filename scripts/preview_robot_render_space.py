#!/usr/bin/env python3
"""Preview legacy and continuous robot render spaces without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_benchmark.config import ROBOT_CONCEPTS
from concept_benchmark.synthetic.helper.robot_catalog import build_robot_instance_catalog
from concept_benchmark.synthetic.helper.robot_draw import draw_robot, render_state_from_metadata


SINGLE_SEMANTIC = {k: v[:1] for k, v in ROBOT_CONCEPTS.items()}


def _render_row(row, resolution: int) -> Image.Image:
    state = render_state_from_metadata(row.to_dict())
    return draw_robot(
        filetype="png",
        width=resolution,
        height=resolution,
        render_state=state,
        **row.to_dict(),
    ).to_pil()


def _grid(
    rows,
    *,
    title: str,
    resolution: int,
    cols: int,
    caption_fields: tuple[str, ...],
) -> Image.Image:
    images = [_render_row(row, resolution) for _, row in rows.iterrows()]
    captions = []
    for _, row in rows.iterrows():
        parts = []
        for field in caption_fields:
            value = row.get(field, "")
            if value:
                parts.append(str(value))
        captions.append(" | ".join(parts))

    if not images:
        raise ValueError("no rows to render")

    pad = 12
    caption_h = 30
    title_h = 36
    cols = max(1, cols)
    n = len(images)
    grid_rows = (n + cols - 1) // cols
    canvas = Image.new(
        "RGBA",
        (
            cols * resolution + (cols + 1) * pad,
            title_h + grid_rows * (resolution + caption_h) + (grid_rows + 1) * pad,
        ),
        (255, 255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), title, fill=(0, 0, 0, 255))

    for idx, (img, caption) in enumerate(zip(images, captions)):
        row_idx = idx // cols
        col_idx = idx % cols
        x = pad + col_idx * (resolution + pad)
        y = title_h + pad + row_idx * (resolution + caption_h)
        canvas.paste(img, (x, y))
        draw.text((x, y + resolution + 6), caption[:48], fill=(0, 0, 0, 255))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/robot_preview"),
        help="Directory for preview PNGs.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=96,
        help="Per-robot preview resolution.",
    )
    parser.add_argument(
        "--mode",
        choices=["continuous_light", "continuous_heavy"],
        default="continuous_light",
        help="Continuous render mode to preview.",
    )
    parser.add_argument("--seed", type=int, default=1014)
    parser.add_argument(
        "--same-semantic-count",
        type=int,
        default=8,
        help="How many nuisance variants to render for one semantic identity.",
    )
    parser.add_argument(
        "--semantic-variety-count",
        type=int,
        default=8,
        help="How many semantic identities to preview in continuous mode.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    same_semantic = build_robot_instance_catalog(
        concepts=SINGLE_SEMANTIC,
        num_robots=args.same_semantic_count,
        resolution=args.resolution,
        seed=args.seed,
        render_space_mode=args.mode,
        validate_renders=True,
    )
    same_semantic_img = _grid(
        same_semantic,
        title=f"One semantic robot, many {args.mode} nuisance states",
        resolution=args.resolution,
        cols=min(4, args.same_semantic_count),
        caption_fields=("accepted_render_space_mode", "pose_descriptor"),
    )
    same_semantic_path = args.output_dir / "same_semantic.png"
    same_semantic_img.save(same_semantic_path)

    semantic_variety = build_robot_instance_catalog(
        concepts=ROBOT_CONCEPTS,
        num_robots=args.semantic_variety_count,
        resolution=args.resolution,
        seed=args.seed,
        render_space_mode=args.mode,
        validate_renders=True,
    )
    semantic_variety_img = _grid(
        semantic_variety,
        title=f"Several semantic identities in {args.mode}",
        resolution=args.resolution,
        cols=min(4, args.semantic_variety_count),
        caption_fields=("head_shape", "body_shape", "stance_bucket"),
    )
    semantic_variety_path = args.output_dir / "semantic_variety.png"
    semantic_variety_img.save(semantic_variety_path)

    legacy = build_robot_instance_catalog(
        concepts=SINGLE_SEMANTIC,
        num_robots=1,
        resolution=args.resolution,
        seed=args.seed,
        render_space_mode="legacy",
        validate_renders=False,
    )
    continuous = build_robot_instance_catalog(
        concepts=SINGLE_SEMANTIC,
        num_robots=1,
        resolution=args.resolution,
        seed=args.seed,
        render_space_mode=args.mode,
        validate_renders=True,
    )
    compare = _grid(
        pd.concat([legacy, continuous], ignore_index=True),
        title="Legacy vs. continuous render state",
        resolution=args.resolution,
        cols=2,
        caption_fields=("render_space_mode", "accepted_render_space_mode", "pose_descriptor"),
    )
    compare_path = args.output_dir / "legacy_vs_continuous.png"
    compare.save(compare_path)

    print(f"Wrote {same_semantic_path}")
    print(f"Wrote {semantic_variety_path}")
    print(f"Wrote {compare_path}")


if __name__ == "__main__":
    main()
