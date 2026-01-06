from argparse import ArgumentParser
import subprocess
import sys
import rich
from rich.panel import Panel
from scripts.robot_demo.utils import DEFAULT_ROBOT_SETTINGS, MISSING_PROP
import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent  # assuming scripts/robot_demo structure

def get_script_path(script_name):
    """Get absolute path to script."""
    return str(PIPELINE_DIR / script_name)



def main():

    p = ArgumentParser(description="run the experimental pipeline.")
    p.add_argument(
        "--stages",
        nargs="+",
        #default=["setup", "cbm", "dnn", "intervene"],
        default=["intervene"],
    )
    p.add_argument("--seed", type=int, default=DEFAULT_ROBOT_SETTINGS['seed'])
    p.add_argument("--draw", action="store_true", help="draw robots at setup")
    p.add_argument("--missing", action="store_true", help="use concept missingness")
    p.add_argument("--ignore-errors", action="store_true", help="continue on errors")
    args = p.parse_args()
    args.missing = True

    # setup pipeline tasks
    pipeline = []
    if "setup" in args.stages:
        cmd = f"python {get_script_path('setup_dataset_robot.py')}"
        sub_cmd = cmd + " --subconcept"
        if args.draw:
            cmd += " --draw"

        pipeline.append(cmd)
        pipeline.append(sub_cmd)

    if "cbm" in args.stages:
        # pipeline.append(
        #     f"python {get_script_path('train_cbm.py')} --seed {args.seed}"
        # )
        # pipeline.append(
        #     f"python {get_script_path('train_cbm.py')} --subconcept --seed {args.seed}"
        # )

        if args.missing:
            for concept_missing_mech in ['mcar', 'mnar']:
                pipeline.append(
                    f"python {get_script_path('train_cbm.py')} --concept_missing {MISSING_PROP} "
                    f"--concept_missing_mech {concept_missing_mech} --seed {args.seed}"
                )
                pipeline.append(
                    f"python {get_script_path('train_cbm.py')} --subconcept "
                    f"--concept_missing {MISSING_PROP} --concept_missing_mech {concept_missing_mech} "
                    f"--seed {args.seed}"
                )
            

    if "dnn" in args.stages:
        pipeline.append(
            f"python {get_script_path('train_dnn.py')} --seed {args.seed}"
        )

    if "intervene" in args.stages:
        pipeline.append(
            f"python {get_script_path('intervene.py')} --seed {args.seed}"
        )
        pipeline.append(
            f"python {get_script_path('intervene.py')} --subconcept --seed {args.seed}"
        )

        if args.missing:
            for concept_missing_mech in ['mcar', 'mnar']:
                pipeline.append(
                    f"python {get_script_path('intervene.py')} --concept_missing {MISSING_PROP} "
                    f"--concept_missing_mech {concept_missing_mech} --seed {args.seed}"
                )
                pipeline.append(
                    f"python {get_script_path('intervene.py')} --subconcept "
                    f"--concept_missing {MISSING_PROP} --concept_missing_mech {concept_missing_mech} "
                    f"--seed {args.seed}"
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