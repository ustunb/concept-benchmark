"""Render the test robots that lack a 224 px image (the LLM caches ask Gemini about 224 px renders).

Rebuilds the robot catalog exactly as the 32 px dataset images were drawn and renders every test robot of the
given seeds that is missing from results/paper/images/robot_224px. Spot check: every 50th rendered robot is also
drawn at 32 px and must be pixel-identical to data/robot_images, otherwise nothing is written.

    python scripts/paper/render_robots_224.py --seed-tests results/_incoming/llm_caches_n10/seed_tests.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from concept_benchmark.config import RobotBenchmarkConfig  # noqa: E402
from concept_benchmark.synthetic.robot import catalog as cat  # noqa: E402
from concept_benchmark.synthetic.robot.draw import draw_robot  # noqa: E402

OUT = REPO / "results/paper/images/robot_224px"
SMALL = REPO / "data/robot_images"


def robot_catalog():
    d = RobotBenchmarkConfig(seed=1014, concept_preset="foot_subtypes").to_dict()
    concepts = d["concepts"]
    n_unique = int(np.prod([len(v) for v in concepts.values()]))
    total = d.get("num_robots") or n_unique * d.get("samples_per_instance", 1)
    df = cat.get_robot_catalog_df(concepts=concepts, repetitions=int(np.ceil(float(total) / n_unique)))
    for name, values in concepts.items():
        if len(values) == 1:
            df = df.query("{}=='{}'".format(name, values[0]))
    df = copy.deepcopy(df)
    flip = df["id"].astype(str).map(lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big") & 1)
    df["foot_orientation"] = np.where(flip.astype(int).values == 1, "vertex", "side")
    return df


def features(df, k) -> dict:
    """Catalog row as plain Python values (draw_robot checks `isinstance(color_scheme, int)`)."""
    return {name: (int(v) if isinstance(v, np.integer) else v) for name, v in df.loc[k].items()}


def png(feat, size: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.png"
        draw_robot(filetype="png", width=size, height=size, **feat).export(str(path))
        return np.asarray(Image.open(path).convert("RGBA"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-tests", type=Path, required=True)
    args = ap.parse_args()
    tests = json.loads(args.seed_tests.read_text())
    want = {int(p.split("_")[1].split(".")[0]) for v in tests.values() for p in v["inputs"]}
    todo = sorted(k for k in want if not (OUT / f"robot_{k:03d}.png").exists())
    print(f"{len(want)} test robots, {len(todo)} without a 224 px render", flush=True)
    df = robot_catalog()
    check = todo[::50]
    for k in check:
        if not (png(features(df, k), 32) == np.asarray(Image.open(SMALL / f"robot_{k:03d}.png").convert("RGBA"))).all():
            raise SystemExit(f"robot {k}: 32 px render differs from data/robot_images; catalog mismatch, nothing written")
    print(f"spot check: {len(check)}/{len(check)} 32 px renders identical to the dataset images", flush=True)
    for n, k in enumerate(todo, 1):
        draw_robot(filetype="png", width=224, height=224, **features(df, k)).export(str(OUT / f"robot_{k:03d}.png"))
        if n % 1000 == 0:
            print(f"  rendered {n}/{len(todo)}", flush=True)
    print(f"wrote {len(todo)} images to {OUT.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
