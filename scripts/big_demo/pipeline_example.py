from argparse import ArgumentParser
import subprocess
import sys
import rich
from rich.panel import Panel

def main():

    p = ArgumentParser(description="run the experimental pipeline.")
    p.add_argument("--action_set_name", default="convex")
    p.add_argument("--resp_thresh", default=0.05, type=float)
    p.add_argument(
        "--models", nargs="+", type=str, default=["glmnet"]
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=["setup", "db", "train", "eval"],
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ignore_errors", default=False, action="store_true")
    args = p.parse_args()

    # setup pipeline tasks
    pipeline = []
    if "setup" in args.stages:
        pipeline.append(f"python scripts/twitter_demo/setup_dataset_actionset_twitterbot.py")

    if "db" in args.stages:
        if args.action_set_name != "immutable": # no need to run this for immutable
            pipeline.append(
                f"python scripts/twitter_demo/sample_reachable_set.py --action_set_name={args.action_set_name} --resp_thresh={args.resp_thresh}"
            )

    if "train" in args.stages:
        for model in args.models:
            if model == "glmnet":
                pipeline.append(
                    f"python scripts/twitter_demo/train_glmnet.py --action_set_name={args.action_set_name}"
                )
            elif model == "xgb":
                pass
            else:
                raise ValueError(f"Unknown model: {model}")

    if "eval" in args.stages:
        for model in args.models:
            pipeline.append(
                f"python scripts/twitter_demo/evaluate_model.py --action_set_name={args.action_set_name} --model_type={model} --resp_thresh={args.resp_thresh}"
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
