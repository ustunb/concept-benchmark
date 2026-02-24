"""LLM client for concept intervention judgments.

Supports Gemini, OpenAI, and Anthropic providers. Used by the LLM and CLIP
intervention regimes to get machine judgments about concept presence in images.
"""
from __future__ import annotations

__all__ = ["make_llm_client", "judge_concept", "judge_concepts_batch"]

import base64
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence


class _LLMBase:
    """Base class for LLM providers."""

    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self.api_key = api_key

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
    def create(cls, name: str, model_name: str, api_key: str) -> _LLMBase:
        kls = cls._providers.get(name.lower())
        if not kls:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ValueError(
                f"Unsupported LLM provider: {name!r}. Available: {available}"
            )
        return kls(model_name, api_key)


def _encode_images_b64(image_paths: Sequence[str | Path]) -> List[tuple]:
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
            continue  # skip unreadable image files
    return out


@_LLMRegistry.register("gemini")
class _GeminiClient(_LLMBase):
    def generate(self, prompt: str, image_paths: Sequence[str | Path] = ()) -> str:
        try:
            import google.generativeai as genai
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "google-generativeai and pillow are required for Gemini. "
                "Install with: pip install google-generativeai pillow"
            ) from e
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        imgs = []
        for p in image_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except (OSError, ValueError):
                pass  # skip unreadable image files
        parts = [prompt] + imgs if imgs else [prompt]
        resp = model.generate_content(
            parts,
            generation_config={"max_output_tokens": 65536},
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
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
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
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
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
    api_key: Optional[str] = None,
    api_key_env: str = "",
) -> _LLMBase:
    """Create an LLM client for the given provider.

    Args:
        provider: One of "gemini", "openai", "anthropic".
        model: Model name/ID for the provider.
        api_key: API key. If None, reads from api_key_env environment variable.
        api_key_env: Environment variable name containing the API key.

    Returns:
        An LLM client with a .generate(prompt, image_paths) method.
    """
    if api_key is None:
        api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(
            f"No API key provided. Set the {api_key_env!r} environment variable "
            f"or pass api_key directly."
        )
    return _LLMRegistry.create(provider, model, api_key)


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
    cache_path: Optional[str | Path] = None,
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
                print(f"  LLM judgments: {done}/{total_calls}")

    # Save cache
    if cache_path is not None:
        cache_p = Path(cache_path)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_p, "w") as f:
            json.dump(cache, f)

    return result
