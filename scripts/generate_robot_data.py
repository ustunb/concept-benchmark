"""Generate a robot classification dataset using RobotDatasetGenerator.

Usage:
    python scripts/generate_robot_data.py --seed 1014
    python scripts/generate_robot_data.py --seed 1014 --subconcept --no-draw
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
    parser.add_argument("--subconcept", action="store_true", help="Use 12 subconcepts instead of 7 ideal")
    parser.add_argument("--no-draw", action="store_true", help="Skip rendering robot PNGs (faster)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for the dataset pickle")
    args = parser.parse_args(argv)

    from concept_benchmark import RobotDatasetGenerator
    from concept_benchmark.ext.fileutils import save

    gen = RobotDatasetGenerator(
        seed=args.seed,
        subconcept=args.subconcept,
        draw=not args.no_draw,
    )
    logger.info(
        "Generating robot dataset: seed=%d, subconcept=%s, draw=%s",
        args.seed, args.subconcept, not args.no_draw,
    )
    dataset = gen.generate()

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
