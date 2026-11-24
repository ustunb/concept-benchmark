"""The paper's default automated concept labels and intervention answers for every robot.

Both come with the package, so neither needs CLIP, an LLM, or the repository:

- :func:`ai_concepts` gives the labels of the three automated concept sets of the paper
  (``machine_labeled``, ``llm_written``, ``clip_words``), from the label-free models of seed 1014.
- :func:`intervention_answers` gives the answers of the paper's interveners (``perfect``, ``expert``,
  ``llm``) for the concepts of a dataset.

Both work on any split of the image robots, e.g. ``DatasetGenerator(...).generate_splits().test``.
To label your own concept list or to ask another LLM, use the pipelines in the repository.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "defaults"
AI_CONCEPT_SOURCES = {
    "machine_labeled": "machine",
    "llm_written": "llm",
    "clip_words": "clip",
}
INTERVENTION_SOURCES = ("perfect", "expert", "llm")
LLM_QUESTION_LISTS = ("true", "human", "llm", "clip")


@cache
def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(DEFAULTS_DIR / name) as data:
        return {k: data[k] for k in data.files}


def _image_rows(dataset) -> np.ndarray:
    """Catalog index of each robot image of ``dataset`` (``robot_123.png`` -> 123)."""
    return np.array(
        [int(Path(str(p)).stem.split("_")[1]) for p in dataset.inputs], dtype=int
    )


def ai_concepts(dataset, source: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Labels of an automated concept set for the robots of ``dataset``.

    Args:
        dataset: image robots, e.g. ``data.train`` of ``DatasetGenerator(...).generate_splits()``.
        source: ``"machine_labeled"`` (the 12 discovered concepts labeled by CLIP),
            ``"llm_written"`` (12 concepts written by Gemini 2.5 Pro), or ``"clip_words"``
            (12 single words ranked by CLIP).

    Returns:
        ``(labels, concept_names, concept_texts)``: ``labels`` has one row per robot of
        ``dataset`` and one 0/1 column per concept.
    """
    if source not in AI_CONCEPT_SOURCES:
        raise ValueError(
            f"source must be one of {sorted(AI_CONCEPT_SOURCES)}, got {source!r}"
        )
    key = AI_CONCEPT_SOURCES[source]
    data = _load("robot_ai_concepts.npz")
    labels = data[f"{key}_labels"][_image_rows(dataset)].astype(int)
    return labels, data[f"{key}_concepts"].tolist(), data[f"{key}_texts"].tolist()


def _llm_question_list(concepts: list[str]) -> str:
    data = _load("robot_llm_answers.npz")
    for questions in LLM_QUESTION_LISTS:
        if set(concepts) <= set(data[f"{questions}_concepts"].tolist()):
            return questions
    raise ValueError(
        "The LLM answered only the concept sets of the paper (true, discovered, llm_written, "
        f"clip_words); these concepts are not one of them: {concepts}"
    )


def intervention_answers(
    dataset,
    source: str,
    *,
    concepts: list[str] | None = None,
    accuracy: float = 0.8,
    seed: int = 0,
) -> np.ndarray:
    """Answers of an intervener for every robot and concept of ``dataset``.

    Args:
        dataset: image robots, e.g. ``data.test`` of ``DatasetGenerator(...).generate_splits()``.
        source: ``"perfect"`` (the true labels), ``"expert"`` (the true labels, each wrong with
            probability ``1 - accuracy``), or ``"llm"`` (Gemini 2.5 Flash-Lite looking at a 224 px
            image of the robot).
        concepts: concept names, in column order; defaults to ``dataset.concepts``. For an
            automated concept set pass its names from :func:`ai_concepts`.
        accuracy: accuracy of the expert (the paper uses 0.8).
        seed: seed of the expert's mistakes.

    Returns:
        A 0/1 array with one row per robot and one column per concept. ``perfect`` and ``expert``
        need the true labels of the concepts, so they work for the true and discovered concept sets.
    """
    if source not in INTERVENTION_SOURCES:
        raise ValueError(
            f"source must be one of {list(INTERVENTION_SOURCES)}, got {source!r}"
        )
    concepts = list(dataset.concepts if concepts is None else concepts)
    if source == "llm":
        questions = _llm_question_list(concepts)
        data = _load("robot_llm_answers.npz")
        names = data[f"{questions}_concepts"].tolist()
        columns = [names.index(c) for c in concepts]
        return data[f"{questions}_answers"][_image_rows(dataset)][:, columns].astype(
            int
        )
    unknown = [c for c in concepts if c not in dataset.concepts]
    if unknown:
        raise ValueError(
            f"{source!r} answers need the true labels, which the dataset has only for "
            f"its own concepts; these are not among them: {unknown}"
        )
    columns = [list(dataset.concepts).index(c) for c in concepts]
    truth = np.asarray(dataset.C, dtype=int)[:, columns]
    if source == "perfect":
        return truth
    mistakes = np.random.default_rng(seed).random(truth.shape) < 1 - accuracy
    return np.where(mistakes, 1 - truth, truth)
