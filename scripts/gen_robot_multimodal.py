
from __future__ import annotations
import argparse, json
from pathlib import Path
from concept_benchmark.synthetic.robot_mm import create_multimodal_robot_dataset, DEFAULT_CONCEPTS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["complete_both","complete_union","incomplete_union"], default="complete_both")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="results/robots_mm")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--color_mode", choices=["color","greyscale"], default="color")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--missing_rate", type=float, default=0.2)
    p.add_argument("--p_overlap", type=float, default=0.3)
    args = p.parse_args()
    out = create_multimodal_robot_dataset(
        mode=args.mode,
        n=args.n,
        concepts=DEFAULT_CONCEPTS,
        seed=args.seed,
        out_dir=args.out_dir,
        image_size=args.image_size,
        color_mode=args.color_mode,
        missing_rate=args.missing_rate,
        p_overlap=args.p_overlap,
    )
    print(json.dumps({"image_csv": str(out.image_csv), "text_csv": str(out.text_csv), "pairs_csv": str(out.pairs_csv)}, indent=2))

if __name__ == "__main__":
    main()
