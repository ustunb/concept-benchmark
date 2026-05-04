"""Tests for the llm_client interface contract, not provider implementations.

All LLM calls are fully mocked (no network access). These tests verify:
- Provider registration and factory validation (make_llm_client).
- The generate() return type contract (_LLMBase interface).
- The judge_concept() parsing logic (YES/NO/ambiguous responses).

Provider-specific behavior (e.g. Gemini API details) is NOT tested here.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

import experiments.llm_client as llm_client_mod
from experiments.llm_client import (
    _LLMBase,
    _LLMRegistry,
    is_local_exec_provider,
    is_retryable_llm_error,
    judge_concept,
    make_llm_client,
)


class TestMakeLLMClient:
    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            make_llm_client("nonexistent_provider", "model-v1", api_key="key123")

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="No API key"):
            make_llm_client("gemini", "model-v1")

    def test_local_exec_provider_does_not_require_api_key(self, monkeypatch):
        monkeypatch.setattr(llm_client_mod.shutil, "which", lambda _: "/usr/bin/fake")
        client = make_llm_client("codex_exec", "gpt-5")
        assert isinstance(client, _LLMBase)

    def test_gemini_provider_registered(self):
        assert "gemini" in _LLMRegistry._providers

    def test_exec_providers_recognized(self):
        assert is_local_exec_provider("codex_exec")
        assert is_local_exec_provider("codex")
        assert is_local_exec_provider("claude_exec")


class TestLLMClientGenerate:
    def test_generate_returns_string(self):
        mock_client = MagicMock(spec=_LLMBase)
        mock_client.generate.return_value = "YES"
        result = mock_client.generate("Is this a robot?", [])
        assert isinstance(result, str)
        assert result == "YES"


class TestJudgeConcept:
    def test_yes_returns_one(self):
        mock_client = MagicMock(spec=_LLMBase)
        mock_client.generate.return_value = "YES"
        val = judge_concept(mock_client, "/fake/path.png", "has red body")
        assert val == 1.0

    def test_no_returns_zero(self):
        mock_client = MagicMock(spec=_LLMBase)
        mock_client.generate.return_value = "NO"
        val = judge_concept(mock_client, "/fake/path.png", "has red body")
        assert val == 0.0

    def test_ambiguous_returns_zero(self):
        mock_client = MagicMock(spec=_LLMBase)
        mock_client.generate.return_value = "Maybe"
        val = judge_concept(mock_client, "/fake/path.png", "has red body")
        assert val == 0.0

    def test_yes_prefix(self):
        mock_client = MagicMock(spec=_LLMBase)
        mock_client.generate.return_value = "YES, I can see it clearly."
        val = judge_concept(mock_client, "/fake/path.png", "has red body")
        assert val == 1.0


class TestRetryableErrors:
    def test_retryable_status_code(self):
        class SomeAPIError(Exception):
            status_code = 429

        assert is_retryable_llm_error("openai", SomeAPIError())

    def test_retryable_openai_error_name(self):
        class RateLimitError(Exception):
            pass

        assert is_retryable_llm_error("openai", RateLimitError())

    def test_retryable_gemini_error_name(self):
        class ResourceExhausted(Exception):
            pass

        assert is_retryable_llm_error("gemini", ResourceExhausted())

    def test_timeout_is_retryable(self):
        err = subprocess.TimeoutExpired(cmd=["codex"], timeout=10)
        assert is_retryable_llm_error("codex_exec", err)

    def test_non_retryable_error(self):
        class BadRequestError(Exception):
            status_code = 400

        assert not is_retryable_llm_error("openai", BadRequestError())


class TestExecLoopClients:
    def test_codex_exec_builds_image_command(self, monkeypatch, tmp_path):
        calls = {}

        def fake_which(name):
            return f"/usr/bin/{name}"

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            out_idx = cmd.index("-o") + 1
            out_path = cmd[out_idx]
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("YES")

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        monkeypatch.setattr(llm_client_mod.shutil, "which", fake_which)
        monkeypatch.setattr(llm_client_mod.subprocess, "run", fake_run)
        client = make_llm_client(
            "codex_exec",
            "gpt-5.4",
            reasoning_effort="medium",
        )
        result = client.generate("Answer yes or no.", [tmp_path / "robot.png"])
        assert result == "YES"
        assert calls["cmd"][:2] == ["/usr/bin/codex", "exec"]
        config_idx = calls["cmd"].index("-c") + 1
        assert calls["cmd"][config_idx] == 'model_reasoning_effort="medium"'
        assert "-i" in calls["cmd"]
        assert str(tmp_path / "robot.png") in calls["cmd"]

    def test_claude_exec_includes_local_paths_in_prompt(self, monkeypatch, tmp_path):
        calls = {}

        def fake_which(name):
            return f"/usr/bin/{name}"

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd

            class Result:
                returncode = 0
                stdout = "NO"
                stderr = ""

            return Result()

        monkeypatch.setattr(llm_client_mod.shutil, "which", fake_which)
        monkeypatch.setattr(llm_client_mod.subprocess, "run", fake_run)
        client = make_llm_client("claude_exec", "sonnet")
        image_path = tmp_path / "robot.png"
        result = client.generate("Answer yes or no.", [image_path])
        assert result == "NO"
        assert calls["cmd"][0] == "/usr/bin/claude"
        assert "--tools" in calls["cmd"]
        assert str(image_path.resolve()) in calls["cmd"][-1]
