"""Enumerate big demo pipeline command combinations.

This helper script produces ready-to-run CLI commands for dataset generation
and model training across the big demo components.  It mirrors the canonical
configuration knobs (concept noise, target accuracy, concept missingness)
so that experiments can be scheduled or parallelized easily.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import types
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import utils as big_demo_utils
except ModuleNotFoundError as err:
    if err.name != "torch":
        raise

    class _TorchCudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    class _TorchMPSStub:
        @staticmethod
        def is_available() -> bool:
            return False

    class _TorchBackendsStub:
        mps = _TorchMPSStub()

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = _TorchCudaStub()
    torch_stub.backends = _TorchBackendsStub()

    def _device(name: str) -> str:
        return name

    torch_stub.device = _device
    sys.modules.setdefault("torch", torch_stub)
    import utils as big_demo_utils

CONCEPT_NOISE: float = big_demo_utils.CONCEPT_NOISE
DIFFICULTY: Dict[str, float] = big_demo_utils.DIFFICULTY


# utils.py currently misspells this constant; fall back gracefully if needed.
CONCEPT_MISSING_RATE: float = getattr(
    big_demo_utils,
    "CONCEPT_MISSING",
    getattr(big_demo_utils, "CONCET_MISSING", 0.05),
)


@dataclass(frozen=True)
class Option:
    """Readable wrapper for sweep values."""

    label: str
    value: Any

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.label


@dataclass
class Combination:
    """Concrete argument set attached to a stage."""

    args: Dict[str, Any]
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class StageConfig:
    name: str
    script: str
    base_args: Mapping[str, Any]
    sweeps: Mapping[str, Sequence[Any]]
    adjust: Callable[[MutableMapping[str, Any], Dict[str, str]], None] | None = None

    def expand(self) -> List[Combination]:
        keys = list(self.sweeps.keys())
        combos: List[Combination] = []
        if not keys:
            combo_args = dict(self.base_args)
            combo_labels: Dict[str, str] = {}
            if self.adjust is not None:
                self.adjust(combo_args, combo_labels)
            combos.append(Combination(combo_args, combo_labels))
            return combos

        sweep_values = [self.sweeps[key] for key in keys]
        for option_tuple in product(*sweep_values):
            combo_args = dict(self.base_args)
            combo_labels: Dict[str, str] = {}
            for key, selected in zip(keys, option_tuple):
                if isinstance(selected, Option):
                    combo_args[key] = selected.value
                    combo_labels[key] = selected.label
                else:
                    combo_args[key] = selected
            if self.adjust is not None:
                self.adjust(combo_args, combo_labels)
            combos.append(Combination(combo_args, combo_labels))
        return combos


TARGET_ACCURACY_OPTIONS: Tuple[Option, ...] = tuple(
    Option(label=label, value=value) for label, value in DIFFICULTY.items()
)

CONCEPT_NOISE_OPTIONS: Tuple[Option, ...] = (
    Option(label="off", value=0.0),
    Option(label="on", value=CONCEPT_NOISE),
)

MISSING_MECHANISMS: Tuple[Option, ...] = (
    Option(label="none", value="none"),
    Option(label="mcar", value="mcar"),
    Option(label="mnar", value="mnar"),
)


def set_missing_rate(args: MutableMapping[str, Any], _: Dict[str, str]) -> None:
    mechanism = args.get("concept_missing_mech", "none")
    args["concept_missing"] = 0.0 if mechanism == "none" else CONCEPT_MISSING_RATE


def build_stage_configs() -> Dict[str, StageConfig]:
    python_prefix = "scripts/big_demo"
    return {
        "dataset_sudoku": StageConfig(
            name="dataset_sudoku",
            script=f"{python_prefix}/setup_sudoku_dataset.py",
            base_args={
                "data_type": "tabular",
                "n": 3,
                "max_corrupt": 21,
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "target_accuracy": TARGET_ACCURACY_OPTIONS,
            },
        ),
        "dataset_robot": StageConfig(
            name="dataset_robot",
            script=f"{python_prefix}/setup_robot_dataset.py",
            base_args={
                "data_type": "image",
                "n": 1,
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "target_accuracy": TARGET_ACCURACY_OPTIONS,
            },
        ),
        "train_dnn_sudoku": StageConfig(
            name="train_dnn_sudoku",
            script=f"{python_prefix}/train_dnn.py",
            base_args={
                "data_name": "sudoku",
                "n": 3,
            },
            sweeps={
                "target_accuracy": TARGET_ACCURACY_OPTIONS,
            },
        ),
        "train_dnn_robot": StageConfig(
            name="train_dnn_robot",
            script=f"{python_prefix}/train_dnn.py",
            base_args={
                "data_name": "robot",
                "data_type": "image",
                "n": 1,
            },
            sweeps={
                "target_accuracy": TARGET_ACCURACY_OPTIONS,
            },
        ),
        "train_concept_detector_sudoku": StageConfig(
            name="train_concept_detector_sudoku",
            script=f"{python_prefix}/train_concept_detectors.py",
            base_args={
                "data_name": "sudoku",
                "n": 3,
                "max_corrupt": 21,
                "concept_missing": CONCEPT_MISSING_RATE,
                "concept_missing_mech": "mcar",
                "target_accuracy": DIFFICULTY["easy"],
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "concept_missing_mech": MISSING_MECHANISMS,
            },
            adjust=set_missing_rate,
        ),
        "train_concept_detector_robot": StageConfig(
            name="train_concept_detector_robot",
            script=f"{python_prefix}/train_concept_detectors.py",
            base_args={
                "data_name": "robot",
                "data_type": "image",
                "n": 1,
                "concept_missing": CONCEPT_MISSING_RATE,
                "concept_missing_mech": "mcar",
                "target_accuracy": DIFFICULTY["easy"],
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "concept_missing_mech": MISSING_MECHANISMS,
            },
            adjust=set_missing_rate,
        ),
        "train_front_end_sudoku": StageConfig(
            name="train_front_end_sudoku",
            script=f"{python_prefix}/train_front_end.py",
            base_args={
                "data_name": "sudoku",
                "n": 3,
                "concept_missing": CONCEPT_MISSING_RATE,
                "concept_missing_mech": "mcar",
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "concept_missing_mech": MISSING_MECHANISMS,
                "target_accuracy": TARGET_ACCURACY_OPTIONS,
            },
            adjust=set_missing_rate,
        ),
        "train_front_end_robot": StageConfig(
            name="train_front_end_robot",
            script=f"{python_prefix}/train_front_end.py",
            base_args={
                "data_name": "robot",
                "data_type": "image",
                "n": 1,
                "concept_missing": CONCEPT_MISSING_RATE,
                "concept_missing_mech": "mcar",
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "concept_missing_mech": MISSING_MECHANISMS,
                "target_accuracy": TARGET_ACCURACY_OPTIONS,
            },
            adjust=set_missing_rate,
        ),
    }


def build_command_parts(script: str, combo: Combination, interpreter: str) -> List[str]:
    parts = [interpreter, script]
    for key, value in combo.args.items():
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
            continue
        parts.append(f"--{key}={value}")
    return parts


def format_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def plain_output(stage_name: str, commands: List[str], combos: List[Combination]) -> str:
    header = f"[{stage_name}]"
    lines = [header]
    for command, combo in zip(commands, combos):
        extra = ""
        if combo.labels:
            label_bits = ", ".join(f"{k}={v}" for k, v in combo.labels.items())
            extra = f"  # {label_bits}"
        lines.append(f"  {command}{extra}")
    return "\n".join(lines)


def json_output(stage_name: str, commands: List[str], combos: List[Combination]) -> Dict[str, Any]:
    stage_payload = []
    for command, combo in zip(commands, combos):
        stage_payload.append(
            {
                "command": command,
                "args": combo.args,
                "labels": combo.labels,
            }
        )
    return {stage_name: stage_payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enumerate big demo pipeline commands")
    all_stages = build_stage_configs()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=sorted(all_stages.keys()),
        default=list(all_stages.keys()),
        help="Subset of stages to enumerate",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "json"),
        default="plain",
        help="Output formatting",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter prefix to use when composing commands",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the output (defaults to stdout)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Only print combination counts instead of full command list",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run each command sequentially after listing them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them (default)",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Continue executing commands even if a prior command fails",
    )
    args = parser.parse_args(argv)

    if args.count and args.execute:
        parser.error("--execute cannot be combined with --count")
    if args.execute and args.dry_run:
        parser.error("Use only one of --execute or --dry-run")

    selected = [all_stages[name] for name in args.stages]

    run_commands = args.execute

    if args.format == "json":
        payload: Dict[str, Any] = {}
        execution_plan: List[Tuple[str, List[str], List[List[str]]]] = []
        for stage in selected:
            combos = stage.expand()
            if args.count:
                payload[stage.name] = {"count": len(combos)}
                continue
            parts_list = [build_command_parts(stage.script, combo, args.python) for combo in combos]
            commands = [format_command(parts) for parts in parts_list]
            payload.update(json_output(stage.name, commands, combos))
            if run_commands:
                execution_plan.append((stage.name, commands, parts_list))
        output_text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        segments: List[str] = []
        execution_plan = []
        for stage in selected:
            combos = stage.expand()
            if args.count:
                segments.append(f"[{stage.name}] {len(combos)} combinations")
                continue
            parts_list = [build_command_parts(stage.script, combo, args.python) for combo in combos]
            commands = [format_command(parts) for parts in parts_list]
            segments.append(plain_output(stage.name, commands, combos))
            if run_commands:
                execution_plan.append((stage.name, commands, parts_list))
        output_text = "\n\n".join(segments)

    if args.output:
        args.output.write_text(output_text)
    else:
        print(output_text)

    if run_commands:
        for stage_name, command_strings, parts_list in execution_plan:
            print(f"\n[execute] {stage_name}")
            for command_string, parts in zip(command_strings, parts_list):
                print(f"  → {command_string}")
                try:
                    subprocess.run(parts, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"    failed with return code {exc.returncode}")
                    if not args.ignore_errors:
                        return exc.returncode or 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
