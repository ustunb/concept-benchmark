"""Generate a Sudoku validation dataset using SudokuDatasetGenerator.

Usage:
    python scripts/generate_sudoku_data.py --seed 171
    python scripts/generate_sudoku_data.py --seed 171 --n-samples 500 --output data/my_sudoku
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_sudoku_data")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a Sudoku validation dataset.")
    parser.add_argument("--seed", type=int, default=171, help="Random seed (default: 171)")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of boards (default: 1000)")
    parser.add_argument("--max-corrupt", type=int, default=9, help="Max cell swaps for invalid boards (default: 9)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for the dataset pickle")
    args = parser.parse_args(argv)

    from concept_benchmark import SudokuDatasetGenerator
    from concept_benchmark.ext.fileutils import save

    gen = SudokuDatasetGenerator(
        seed=args.seed,
        n_samples=args.n_samples,
        max_corrupt=args.max_corrupt,
    )
    logger.info(
        "Generating sudoku dataset: seed=%d, n_samples=%d, max_corrupt=%d",
        args.seed, args.n_samples, args.max_corrupt,
    )
    dataset = gen.generate()

    logger.info(
        "Dataset: %d train, %d val, %d test, %d concepts",
        dataset.training.n, dataset.validation.n, dataset.test.n, dataset.training.C.shape[1],
    )

    if args.output:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = out_path / "sudoku_dataset.pkl"
        save(dataset, save_path, overwrite=True)
        logger.info("Saved to %s", save_path)
    else:
        logger.info("No --output specified; dataset generated but not saved to disk.")
        logger.info("Use --output <dir> to save.")


if __name__ == "__main__":
    main()
