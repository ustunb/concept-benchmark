from __future__ import annotations

from pathlib import Path

import pytest

from concept_benchmark.config import RobotBenchmarkConfig
from scripts.robot_pipeline import (
    AUTOMATED_REGIMES,
    _automated_intervention_llm_settings,
    _parse_args,
    _resolve_automated_regime_concepts_file,
)


def test_custom_is_registered():
    assert "custom" in AUTOMATED_REGIMES


def test_llm_uses_packaged_default_concepts_file():
    cfg = RobotBenchmarkConfig()
    path = _resolve_automated_regime_concepts_file(cfg, "llm")
    assert path.name == "llm.jsonl"
    assert path.exists()


def test_custom_requires_explicit_concepts_file():
    cfg = RobotBenchmarkConfig(intervention_regimes=["baseline", "custom"])
    with pytest.raises(ValueError, match="config.custom_concepts_file"):
        _resolve_automated_regime_concepts_file(cfg, "custom")


def test_custom_rejects_missing_concepts_path(tmp_path):
    cfg = RobotBenchmarkConfig(
        custom_concepts_file=str(tmp_path / "missing_custom.jsonl")
    )
    with pytest.raises(ValueError, match="does not exist"):
        _resolve_automated_regime_concepts_file(cfg, "custom")


def test_custom_accepts_explicit_concepts_file(tmp_path):
    concepts = tmp_path / "custom.jsonl"
    concepts.write_text('{"key":"demo","text":"demo"}\n')
    cfg = RobotBenchmarkConfig(custom_concepts_file=str(concepts))
    path = _resolve_automated_regime_concepts_file(cfg, "custom")
    assert path == concepts.resolve()


def test_automated_intervention_settings_use_gemini_defaults():
    cfg = RobotBenchmarkConfig()
    settings = _automated_intervention_llm_settings(cfg)
    assert settings == {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "reasoning_effort": "",
        "api_key": "",
        "api_key_env": "GEMINI_API_KEY",
        "batch_size": 100,
        "batch_sleep": 5.0,
        "workers": 1,
        "cache_all_concepts": False,
    }


def test_automated_intervention_settings_use_exec_loop_batch_size():
    cfg = RobotBenchmarkConfig(llm_provider="codex_exec")
    settings = _automated_intervention_llm_settings(cfg)
    assert settings["batch_size"] == 4
    assert settings["batch_sleep"] == 0.0
    assert settings["workers"] == 1


def test_automated_intervention_settings_honor_runtime_overrides():
    cfg = RobotBenchmarkConfig(
        llm_provider="codex_exec",
        llm_workers=4,
        llm_batch_size=8,
        llm_batch_sleep=1.5,
    )
    settings = _automated_intervention_llm_settings(cfg)
    assert settings["workers"] == 4
    assert settings["batch_size"] == 8
    assert settings["batch_sleep"] == 1.5


def test_parse_args_accepts_new_automated_regime_flags(tmp_path):
    placeholder = tmp_path / "custom.jsonl"
    args = _parse_args(
        [
            "--regimes",
            "baseline",
            "custom",
            "--llm-provider",
            "codex_exec",
            "--llm-model",
            "gpt-5",
            "--llm-reasoning-effort",
            "medium",
            "--llm-cache-only",
            "--llm-workers",
            "4",
            "--llm-batch-size",
            "8",
            "--llm-batch-sleep",
            "0",
            "--llm-api-key-env",
            "GEMINI_API_KEY_ALT",
            "--llm-concepts-file",
            "concept_benchmark/concept_descriptions/llm.jsonl",
            "--clip-concepts-file",
            "concept_benchmark/concept_descriptions/clip.jsonl",
            "--custom-concepts-file",
            str(placeholder),
        ]
    )
    assert args.regimes == ["baseline", "custom"]
    assert args.llm_provider == "codex_exec"
    assert args.llm_model == "gpt-5"
    assert args.llm_reasoning_effort == "medium"
    assert args.llm_cache_only is True
    assert args.llm_workers == 4
    assert args.llm_batch_size == 8
    assert args.llm_batch_sleep == 0.0
    assert args.llm_api_key_env == "GEMINI_API_KEY_ALT"
    assert Path(args.custom_concepts_file) == placeholder
