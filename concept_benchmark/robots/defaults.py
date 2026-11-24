"""The paper's default automated concept labels and intervention answers for every robot.

Both come with the package, so neither needs CLIP, an LLM, or the repository:

- :func:`ai_concepts` gives the labels of the three automated concept sets of the paper
  (``machine_labeled``, ``llm_written``, ``clip_words``).
- :func:`intervention_answers` gives the answers of the paper's interveners (``perfect``, ``expert``,
  ``llm``) for the concepts of a dataset.

Both work on any split of the image robots of the default catalog (30,720 renders), e.g.
``DatasetGenerator(...).generate_splits().test``. To label your own concept list or to ask another
LLM, use the pipelines in the repository.
"""

from __future__ import annotations

import hashlib
import zlib
from functools import cache
from pathlib import Path

import numpy as np

DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "defaults"
CATALOG_SIZE = 30720
# Hash of the default catalog's concepts (meta["UC"] and meta["unfiltered_concepts"]); the shipped
# files describe that catalog, row k being the image robot_k.png.
CATALOG_SIGNATURE = "db9769434158adbd"
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


def catalog_signature(meta: dict) -> str:
    """Hash of the robot catalog a dataset was generated from."""
    concepts = np.ascontiguousarray(meta["UC"], dtype=np.uint8).tobytes()
    names = "|".join(map(str, meta["unfiltered_concepts"])).encode()
    return hashlib.sha256(concepts + names).hexdigest()[:16]


def image_index(path: str | Path) -> int:
    """Catalog index of a robot image (``robot_123.png`` -> 123)."""
    return int(Path(str(path)).stem.split("_")[1])


def _image_rows(dataset) -> np.ndarray:
    # a split keeps only its own rows of the catalog; the full dataset has all of them
    meta = getattr(dataset, "parent", dataset).meta
    if not (
        isinstance(meta, dict)
        and "UC" in meta
        and catalog_signature(meta) == CATALOG_SIGNATURE
    ):
        raise ValueError(
            "The default annotations and answers cover the image robots of the default catalog "
            "(30,720 renders); this dataset was generated from another catalog or modality."
        )
    return np.array([image_index(p) for p in dataset.inputs], dtype=int)


def ai_concepts(dataset, source: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Labels of an automated concept set for the robots of ``dataset``.

    The labels come from the paper's label-free models of seed 1014, applied to every robot image.

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


def _llm_question_list(concepts: list[str], dataset_concepts: list[str]) -> str:
    """The question list the LLM answered for ``concepts``.

    The dataset's own concept set decides when a list covers it, so a subset of the discovered
    concepts gets the discovered questions even where the true concepts share a name.
    """
    data = _load("robot_llm_answers.npz")
    candidates = [concepts]
    if set(concepts) <= set(dataset_concepts):
        candidates.insert(0, dataset_concepts)
    for names in candidates:
        for questions in LLM_QUESTION_LISTS:
            if set(names) <= set(data[f"{questions}_concepts"].tolist()):
                return questions
    raise ValueError(
        "The LLM answered only the concept sets of the paper (true, discovered, llm_written, "
        f"clip_words); these concepts are not one of them: {concepts}"
    )


def _expert_mistakes(
    rows: np.ndarray, concepts: list[str], accuracy: float, seed: int
) -> np.ndarray:
    """Which answers the expert gets wrong, fixed per robot image and concept for a given seed."""
    draws = [
        np.random.default_rng([seed, zlib.crc32(c.encode())]).random(CATALOG_SIZE)[rows]
        for c in concepts
    ]
    return np.stack(draws, axis=1) < 1 - accuracy


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
        accuracy: accuracy of the expert, between 0 and 1 (the paper uses 0.8).
        seed: seed of the expert's mistakes; for a given seed, each robot image and concept always
            gets the same answer, whatever the split or the other concepts.

    Returns:
        A 0/1 array with one row per robot and one column per concept. ``perfect`` and ``expert``
        need the true labels of the concepts, so they work for the true and discovered concept sets.
    """
    if source not in INTERVENTION_SOURCES:
        raise ValueError(
            f"source must be one of {list(INTERVENTION_SOURCES)}, got {source!r}"
        )
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be between 0 and 1, got {accuracy}")
    concepts = list(dataset.concepts if concepts is None else concepts)
    rows = _image_rows(dataset)
    if source == "llm":
        questions = _llm_question_list(concepts, list(dataset.concepts))
        data = _load("robot_llm_answers.npz")
        names = data[f"{questions}_concepts"].tolist()
        columns = [names.index(c) for c in concepts]
        return data[f"{questions}_answers"][rows][:, columns].astype(int)
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
    mistakes = _expert_mistakes(rows, concepts, accuracy, seed)
    return np.where(mistakes, 1 - truth, truth)
