"""Enumerate big demo pipeline command combinations.

This helper script produces ready-to-run CLI commands for dataset generation
and model training across the big demo components.  It mirrors the canonical
configuration knobs (concept noise, target accuracy, concept missingness)
so that experiments can be scheduled or parallelized easily.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import types
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence, Tuple

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

CONCEPT_NOISE: Sequence[float] = big_demo_utils.CONCEPT_NOISE
DIFFICULTY: Dict[str, float] = big_demo_utils.DIFFICULTY
CONCEPT_MISSING_RATE: Sequence[float] = big_demo_utils.CONCEPT_MISSING


@dataclass(frozen=True)
class Option:
    """Readable wrapper for sweep values."""

    label: str
    value: Any


@dataclass
class Combination:
    """Concrete argument set attached to a stage."""

    args: Dict[str, Any]
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MissingnessMode:
    mechanism: str
    level: float


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

CONCEPT_NOISE_OPTIONS: Tuple[Option, ...] = tuple(
    Option(label=f"concept_noise_{noise_level}", value=noise_level) for noise_level in CONCEPT_NOISE
)

MISSING_MECHANISMS: Tuple[Option, ...] = (
    Option(label="none", value="none"),
    Option(label="mcar", value="mcar"),
    Option(label="mnar", value="mnar"),
)


def _format_missing_level(level: float) -> str:
    return str(level)


def _build_missingness_modes() -> Tuple[Option, ...]:
    options: List[Option] = []
    for mechanism_option in MISSING_MECHANISMS:
        mechanism = mechanism_option.value
        levels: Sequence[float]
        if mechanism == "none":
            levels = (0.0,)
        else:
            levels = CONCEPT_MISSING_RATE
        for missing_level in levels:
            label_level = _format_missing_level(missing_level)
            options.append(
                Option(
                    label=f"missing_{mechanism}_{label_level}",
                    value=MissingnessMode(mechanism=mechanism, level=missing_level),
                )
            )
    return tuple(options)


MISSINGNESS_MODES: Tuple[Option, ...] = _build_missingness_modes()


def set_missing_rate(args: MutableMapping[str, Any], labels: Dict[str, str]) -> None:
    mode = args.pop("missingness", None)
    if isinstance(mode, MissingnessMode):
        labels.pop("missingness", None)
        args["concept_missing_mech"] = mode.mechanism
        args["concept_missing"] = mode.level
        labels["concept_missing_mech"] = mode.mechanism
        labels["concept_missing"] = f"concept_missing_{_format_missing_level(mode.level)}"
    mechanism = args.get("concept_missing_mech", "none")
    if mechanism == "none":
        args["concept_missing"] = 0.0


def build_stage_configs() -> Dict[str, StageConfig]:
    python_prefix = "scripts/big_demo"
    return {
        "setup_dataset_sudoku": StageConfig(
            name="setup_dataset_sudoku",
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
        "setup_dataset_robot": StageConfig(
            name="setup_dataset_robot",
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
                "concept_missing": 0.0,
                "concept_missing_mech": "none",
                "target_accuracy": DIFFICULTY["easy"],
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "missingness": MISSINGNESS_MODES,
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
                "concept_missing": 0.0,
                "concept_missing_mech": "none",
                "target_accuracy": DIFFICULTY["easy"],
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "missingness": MISSINGNESS_MODES,
            },
            adjust=set_missing_rate,
        ),
        "train_front_end_sudoku": StageConfig(
            name="train_front_end_sudoku",
            script=f"{python_prefix}/train_front_end.py",
            base_args={
                "data_name": "sudoku",
                "n": 3,
                "max_corrupt": 21
            },
            # Just train one "good" front-end model for sudoku
            sweeps={},
        ),
        "train_front_end_robot": StageConfig(
            name="train_front_end_robot",
            script=f"{python_prefix}/train_front_end.py",
            base_args={
                "data_name": "robot",
                "data_type": "image",
                "n": 1,
                "concept_missing": 0.0,
                "concept_missing_mech": "none",
            },
            sweeps={
                "concept_noise": CONCEPT_NOISE_OPTIONS,
                "missingness": MISSINGNESS_MODES,
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


STAGE_TYPES = [
    "setup_dataset",
    "train_dnn",
    "train_concept_detector",
    "train_front_end"
]

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enumerate big demo pipeline commands")
    all_stages = build_stage_configs()
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=("sudoku", "robot"),
        default=("sudoku", "robot"),
        help="Select dataset stages (implies related training stages)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_TYPES,
        default=STAGE_TYPES,
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
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional log file path for recording pipeline progress",
    )
    args = parser.parse_args(argv)

    if args.count and args.execute:
        parser.error("--execute cannot be combined with --count")
    if args.execute and args.dry_run:
        parser.error("Use only one of --execute or --dry-run")

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        log_path = args.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger(__name__)

    selected_stages = [
        f"{stype}_{dset}" for stype in args.stages for dset in args.dataset
    ]

    selected = [all_stages[name] for name in selected_stages if name in all_stages]

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
            logger.info("Executing %s (%d commands)", stage_name, len(parts_list))
            print(f"\n[execute] {stage_name}")
            for command_string, parts in zip(command_strings, parts_list):
                print(f"  → {command_string}")
                logger.info("Starting command for %s: %s", stage_name, command_string)
                try:
                    subprocess.run(parts, check=True)
                except subprocess.CalledProcessError as exc:
                    logger.error(
                        "Command failed for %s (return code %s): %s",
                        stage_name,
                        exc.returncode,
                        command_string,
                    )
                    print(f"    failed with return code {exc.returncode}")
                    if not args.ignore_errors:
                        return exc.returncode or 1
                    continue
                logger.info("Completed command for %s: %s", stage_name, command_string)
            logger.info("Finished %s", stage_name)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
