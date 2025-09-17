from __future__ import annotations
import argparse, json
from pathlib import Path
from concept_benchmark.synthetic.robot_mm import create_multimodal_robot_dataset, DEFAULT_CONCEPTS

settings = {
    "mode": "complete_both",
    "n": 200,
    "out_dir": "results/robots_mm",
    "image_size": 256,
    "color_mode": "color",
    "seed": 0,
    "missing_rate": 0.2,
    "p_overlap": 0.3,
}

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["complete_both","complete_union","incomplete_union"])
p.add_argument("--n", type=int)
p.add_argument("--out_dir", type=str)
p.add_argument("--image_size", type=int)
p.add_argument("--color_mode", choices=["color","greyscale"])
p.add_argument("--seed", type=int)
p.add_argument("--missing_rate", type=float)
p.add_argument("--p_overlap", type=float)
args, _ = p.parse_known_args()
for k, v in vars(args).items():
    if v is not None:
        settings[k] = v

out_dir = Path(settings["out_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

out = create_multimodal_robot_dataset(
    mode=settings["mode"],
    n=settings["n"],
    concepts=DEFAULT_CONCEPTS,
    seed=settings["seed"],
    out_dir=str(out_dir),
    image_size=settings["image_size"],
    color_mode=settings["color_mode"],
    missing_rate=settings["missing_rate"],
    p_overlap=settings["p_overlap"],
)

print(json.dumps({
    "image_csv": str(out.image_csv),
    "text_csv": str(out.text_csv),
    "pairs_csv": str(out.pairs_csv)
}, indent=2))
