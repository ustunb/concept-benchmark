"""Tests for concept_benchmark.llm_client module (fully mocked, no network)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from experiments.llm_client import (
    _LLMBase,
    _LLMRegistry,
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

    def test_gemini_provider_registered(self):
        assert "gemini" in _LLMRegistry._providers


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
