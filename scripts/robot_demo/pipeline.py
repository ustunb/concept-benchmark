from argparse import ArgumentParser
import itertools
import subprocess
import sys
import rich
from rich.panel import Panel
from scripts.robot_demo.utils import DEFAULT_ROBOT_SETTINGS


def main():

    p = ArgumentParser(description="run the experimental pipeline.")
    p.add_argument(
        "--stages",
        nargs="+",
        default=["setup", "cbm", "dnn", "intervene"],
    )
    p.add_argument("--seed", type=int, default=DEFAULT_ROBOT_SETTINGS['seed'])
    args = p.parse_args()

    # setup pipeline tasks
    pipeline = []
    if "setup" in args.stages:
        pipeline.append("python scripts/robot_demo/setup_dataset_robot.py")
        pipeline.append("python scripts/robot_demo/setup_dataset_robot.py --subconcept")

    if "cbm" in args.stages:
        pipeline.append(
            f"python scripts/robot_demo/train_cbm.py --seed {args.seed}"
        )
        pipeline.append(
            f"python scripts/robot_demo/train_cbm.py --subconcept --seed {args.seed}"
        )

    if "dnn" in args.stages:
        pipeline.append(
            f"python scripts/robot_demo/train_dnn.py --seed {args.seed}"
        )

    if "intervene" in args.stages:
        pipeline.append(
            f"python scripts/robot_demo/intervene.py --seed {args.seed}"
        )
        pipeline.append(
            f"python scripts/robot_demo/intervene.py --subconcept --seed {args.seed}"
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
        except:
            failed = True
            if not args.ignore_errors:
                break

    sys.exit(failed)


if __name__ == "__main__":
    main()