"""LLM client for concept intervention judgments.

Supports API-backed providers (Gemini, OpenAI, Anthropic) plus optional local
CLI loops (Codex exec, Claude print mode). Used by the automated intervention
regimes to get machine judgments about concept presence in images.
"""

from __future__ import annotations

__all__ = [
    "is_retryable_llm_error",
    "is_local_exec_provider",
    "judge_concept",
    "judge_concepts_batch",
    "make_llm_client",
]

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class _LLMBase:
    """Base class for LLM providers."""

    def __init__(
        self,
        model_name: str = "",
        api_key: str = "",
        config_overrides: Sequence[str] = (),
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.config_overrides = tuple(config_overrides)

    def generate(self, prompt: str, image_paths: Sequence[str | Path] = ()) -> str:
        raise NotImplementedError


class _LLMRegistry:
    _providers: dict = {}

    @classmethod
    def register(cls, name: str):
        def deco(kls):
            cls._providers[name.lower()] = kls
            return kls

        return deco

    @classmethod
    def create(
        cls,
        name: str,
        model_name: str,
        api_key: str,
        *,
        config_overrides: Sequence[str] = (),
    ) -> _LLMBase:
        kls = cls._providers.get(name.lower())
        if not kls:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ValueError(
                f"Unsupported LLM provider: {name!r}. Available: {available}"
            )
        return kls(model_name, api_key, config_overrides=config_overrides)


def _encode_images_b64(image_paths: Sequence[str | Path]) -> list[tuple]:
    out = []
    for p in image_paths:
        try:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            ext = Path(p).suffix.lower()
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".webp":
                mime = "image/webp"
            else:
                mime = "image/png"
            out.append((b64, mime))
        except (OSError, ValueError):
            logging.getLogger(__name__).debug("Skipping unreadable image: %s", p)
            continue
    return out


LOCAL_EXEC_PROVIDERS = frozenset({"codex", "codex_exec", "claude_exec"})


def is_local_exec_provider(provider: str) -> bool:
    return provider.strip().lower() in LOCAL_EXEC_PROVIDERS


class _ExecCLIClient(_LLMBase):
    """Base class for local CLI-backed LLM providers."""

    executable_name = ""

    def __init__(
        self,
        model_name: str = "",
        api_key: str = "",
        config_overrides: Sequence[str] = (),
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            config_overrides=config_overrides,
        )
        self.executable_path = shutil.which(self.executable_name)
        if self.executable_path is None:
            raise FileNotFoundError(
                f"{self.executable_name} executable not found on PATH."
            )

    def _build_command(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        output_path: str | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def _prepare_prompt(self, prompt: str, image_paths: Sequence[str | Path]) -> str:
        return prompt

    def generate(self, prompt: str, image_paths: Sequence[str | Path] = ()) -> str:
        prompt_text = self._prepare_prompt(prompt, image_paths)
        output_path = None
        if self.executable_name == "codex":
            fd, output_path = tempfile.mkstemp(prefix="codex_exec_", suffix=".txt")
            os.close(fd)
        cmd = self._build_command(prompt_text, image_paths, output_path=output_path)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"{self.executable_name} exec failed with exit code {result.returncode}: {detail}"
            )
        if output_path is not None:
            try:
                return Path(output_path).read_text(encoding="utf-8").strip()
            finally:
                Path(output_path).unlink(missing_ok=True)
        return (result.stdout or "").strip()


@_LLMRegistry.register("codex")
@_LLMRegistry.register("codex_exec")
class _CodexExecClient(_ExecCLIClient):
    executable_name = "codex"

    def _build_command(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        output_path: str | None = None,
    ) -> list[str]:
        cmd = [
            self.executable_path,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "-C",
            str(Path.cwd()),
        ]
        if self.model_name:
            cmd.extend(["-m", self.model_name])
        for override in self.config_overrides:
            cmd.extend(["-c", override])
        for image_path in image_paths:
            cmd.extend(["-i", str(image_path)])
        if output_path is not None:
            cmd.extend(["-o", output_path])
        cmd.append(prompt)
        return cmd


@_LLMRegistry.register("claude_exec")
class _ClaudeExecClient(_ExecCLIClient):
    executable_name = "claude"

    def _prepare_prompt(self, prompt: str, image_paths: Sequence[str | Path]) -> str:
        if not image_paths:
            return prompt
        image_lines = "\n".join(f"- {Path(p).resolve()}" for p in image_paths)
        return (
            f"{prompt}\n\n"
            "Image files are available locally at the following paths. "
            "Use read-only inspection to analyze them. Do not modify any files.\n"
            f"{image_lines}"
        )

    def _build_command(
        self,
        prompt: str,
        image_paths: Sequence[str | Path],
        output_path: str | None = None,
    ) -> list[str]:
        del image_paths, output_path
        cmd = [
            self.executable_path,
            "-p",
            "--output-format",
            "text",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Bash,Read",
            "--add-dir",
            str(Path.cwd()),
        ]
        if self.model_name:
            cmd.extend(["--model", self.model_name])
        cmd.append(prompt)
        return cmd


@_LLMRegistry.register("gemini")
class _GeminiClient(_LLMBase):
    def generate(self, prompt: str, image_paths: Sequence[str | Path] = ()) -> str:
        try:
            from google import genai
            from google.genai import types
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "google-genai and pillow are required for Gemini. "
                "Install with: pip install google-genai pillow"
            ) from e
        client = genai.Client(api_key=self.api_key)
        imgs = []
        for p in image_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except (OSError, ValueError):
                logging.getLogger(__name__).debug("Skipping unreadable image: %s", p)
        parts = [prompt] + imgs if imgs else [prompt]
        resp = client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=types.GenerateContentConfig(
                max_output_tokens=65536,
            ),
        )
        return (getattr(resp, "text", None) or "").strip()


@_LLMRegistry.register("openai")
class _OpenAIClient(_LLMBase):
    def generate(self, prompt: str, image_paths: Sequence[str | Path] = ()) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK is required. Install with: pip install openai"
            ) from e
        client = OpenAI(api_key=self.api_key)
        content: list = [{"type": "text", "text": prompt}]
        for b64, mime in _encode_images_b64(image_paths):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()


@_LLMRegistry.register("anthropic")
class _AnthropicClient(_LLMBase):
    def generate(self, prompt: str, image_paths: Sequence[str | Path] = ()) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic SDK is required. Install with: pip install anthropic"
            ) from e
        client = anthropic.Anthropic(api_key=self.api_key)
        content: list = [{"type": "text", "text": prompt}]
        for b64, mime in _encode_images_b64(image_paths):
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                }
            )
        resp = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        text_parts = []
        for part in getattr(resp, "content", []) or []:
            if getattr(part, "type", None) == "text":
                text_parts.append(part.text)
        return " ".join(text_parts).strip()


def make_llm_client(
    provider: str,
    model: str,
    api_key: str | None = None,
    api_key_env: str = "",
    reasoning_effort: str = "",
) -> _LLMBase:
    """Create an LLM client for the given provider.

    Args:
        provider: One of "gemini", "openai", "anthropic", "codex",
            "codex_exec", or "claude_exec".
        model: Model name/ID for the provider.
        api_key: API key. If None, reads from api_key_env environment variable.
        api_key_env: Environment variable name containing the API key.

    Returns:
        An LLM client with a .generate(prompt, image_paths) method.
    """
    if is_local_exec_provider(provider):
        config_overrides: list[str] = []
        effort = reasoning_effort.strip()
        if effort and provider.strip().lower() in {"codex", "codex_exec"}:
            config_overrides.append(f'model_reasoning_effort="{effort}"')
        return _LLMRegistry.create(
            provider,
            model,
            api_key or "",
            config_overrides=config_overrides,
        )
    if api_key is None:
        api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(
            f"No API key provided. Set the {api_key_env!r} environment variable "
            f"or pass api_key directly."
        )
    return _LLMRegistry.create(provider, model, api_key)


def is_retryable_llm_error(provider: str, exc: Exception) -> bool:
    """Return whether *exc* is a transient provider error worth retrying."""
    provider_name = provider.strip().lower()
    exc_name = exc.__class__.__name__.lower()
    status_code = getattr(exc, "status_code", None)

    if isinstance(exc, subprocess.TimeoutExpired):
        return True

    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    retryable_names = {
        "gemini": {"resourceexhausted"},
        "openai": {
            "apiconnectionerror",
            "apierror",
            "apitimeouterror",
            "internalservererror",
            "ratelimiterror",
        },
        "anthropic": {
            "anthropicconnectionerror",
            "anthropicerror",
            "apiconnectionerror",
            "apitimeouterror",
            "internalservererror",
            "overloadederror",
            "ratelimiterror",
        },
    }
    return exc_name in retryable_names.get(provider_name, set())


CONCEPT_JUDGMENT_PROMPT = """Look at this robot image. For the following visual concept, answer YES or NO:

Concept: "{concept_text}"

Does this robot exhibit this concept? Reply with exactly one word: YES or NO."""


def judge_concept(
    client: _LLMBase,
    image_path: str | Path,
    concept_text: str,
) -> float:
    """Ask an LLM to judge whether a concept is present in an image.

    Returns 1.0 for YES, 0.0 for NO.
    """
    prompt = CONCEPT_JUDGMENT_PROMPT.format(concept_text=concept_text)
    response = client.generate(prompt, [image_path])
    answer = response.strip().upper()
    if answer.startswith("YES"):
        return 1.0
    return 0.0


def judge_concepts_batch(
    client: _LLMBase,
    image_paths: Sequence[str | Path],
    concept_texts: Sequence[str],
    mask: "np.ndarray",
    cache_path: str | Path | None = None,
) -> "np.ndarray":
    """Judge concepts for a batch of images where mask is True.

    Args:
        client: LLM client.
        image_paths: Paths to images (N,).
        concept_texts: Concept description strings (M,).
        mask: Boolean array (N, M) -- only judge where True.
        cache_path: Optional path to JSON cache for results.

    Returns:
        Array (N, M) with LLM judgments (1.0/0.0) where mask is True,
        NaN elsewhere.
    """
    import numpy as np

    N = len(image_paths)
    M = len(concept_texts)
    result = np.full((N, M), np.nan, dtype=np.float32)

    # Load cache if available
    cache: dict = {}
    if cache_path is not None:
        cache_p = Path(cache_path)
        if cache_p.exists():
            with open(cache_p, "r") as f:
                cache = json.load(f)

    total_calls = int(mask.sum())
    done = 0
    for i in range(N):
        for j in range(M):
            if not mask[i, j]:
                continue
            cache_key = f"{image_paths[i]}::{concept_texts[j]}"
            if cache_key in cache:
                result[i, j] = float(cache[cache_key])
            else:
                val = judge_concept(client, image_paths[i], concept_texts[j])
                result[i, j] = val
                cache[cache_key] = val
            done += 1
            if done % 50 == 0:
                logging.getLogger(__name__).info(
                    "LLM judgments: %d/%d", done, total_calls
                )

    # Save cache
    if cache_path is not None:
        cache_p = Path(cache_path)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_p, "w") as f:
            json.dump(cache, f)

    return result
