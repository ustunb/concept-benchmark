"""Generate a robot classification dataset using DatasetGenerator.

Usage:
    python scripts/generate_robot_data.py --seed 1014
    python scripts/generate_robot_data.py --seed 1014 --concept-preset foot_subtypes --no-draw
    python scripts/generate_robot_data.py --seed 1014 --output data/my_robots
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_robot_data")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a robot classification dataset.")
    parser.add_argument("--seed", type=int, default=1014, help="Random seed (default: 1014)")
    parser.add_argument("--concept-preset", choices=["ground_truth", "foot_subtypes"], default="ground_truth", help="Concept preset: ground_truth (7 ideal) or foot_subtypes (12 subconcepts)")
    parser.add_argument("--no-draw", action="store_true", help="Skip rendering robot PNGs (faster)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for the dataset pickle")
    args = parser.parse_args(argv)

    from concept_benchmark import DatasetGenerator
    from concept_benchmark.ext.fileutils import save

    gen = DatasetGenerator(
        "robot",
        seed=args.seed,
        concept_preset=args.concept_preset,
        render_images=not args.no_draw,
    )
    logger.info(
        "Generating robot dataset: seed=%d, concept_preset=%s, render_images=%s",
        args.seed, args.concept_preset, not args.no_draw,
    )
    dataset = gen.generate()
    from concept_benchmark.config import PRESET_EXCLUDED_CONCEPTS
    dataset.drop_concepts(PRESET_EXCLUDED_CONCEPTS[args.concept_preset])
    dataset.sample(test_size=10000, val_size=0.2, train_size=3800, seed=args.seed)

    logger.info(
        "Dataset: %d train, %d val, %d test, %d concepts",
        dataset.training.n, dataset.validation.n, dataset.test.n, dataset.training.C.shape[1],
    )

    if args.output:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = out_path / "robot_dataset.pkl"
        save(dataset, save_path, overwrite=True)
        logger.info("Saved to %s", save_path)
    else:
        logger.info("No --output specified; dataset generated but not saved to disk.")
        logger.info("Use --output <dir> to save.")


if __name__ == "__main__":
    main()
