from argparse import ArgumentParser
import subprocess
import sys
import rich
from rich.panel import Panel
from scripts.sudoku_demo.utils import DEFAULT_SUDOKU_SETTINGS


def main():

    p = ArgumentParser(description="run the experimental pipeline.")
    p.add_argument(
        "--stages",
        nargs="+",
        default=["setup", "ocr", "cs", "dnn", "intervene"],
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SUDOKU_SETTINGS['seed'])
    p.add_argument("--ignore-errors", action="store_true", help="continue on errors")
    args = p.parse_args()

    # setup pipeline tasks
    pipeline = []
    if "setup" in args.stages:
        pipeline.append("python scripts/sudoku_demo/make_ocr_dataset.py")

    if "ocr" in args.stages:
        pipeline.append(
            f"python scripts/sudoku_demo/train_ocr_fast.py --seed {args.seed}"
        )

    if "cs" in args.stages:
        pipeline.append(
            f"python scripts/sudoku_demo/train_cs.py --seed {args.seed}"
        )

    if "dnn" in args.stages:
        pipeline.append(
            f"python scripts/sudoku_demo/train_dnn.py --seed {args.seed}"
        )

    if "intervene" in args.stages:
        pipeline.append(
            f"python scripts/sudoku_demo/intervene.py --seed {args.seed}"
        )

    # Run each command in the list
    failed = False
    for command in pipeline:
        rich.print(Panel(f"[bold]{command}[/bold]"))
        try:
            subprocess.run(
                command, shell=True, check=True, text=True, capture_output=False
            )
        except KeyboardInterrupt:
            failed = True
            break
        except Exception:
            failed = True
            if not args.ignore_errors:
                break

    sys.exit(failed)


if __name__ == "__main__":
    main()