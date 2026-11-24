"""Every kind of robot dataset × every default-answer call: shapes, values, and one answer per robot.

The same robot image and concept must get the same label or answer whatever dataset, split,
embedding, corruption, column order, or concept subset it is asked through.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from concept_benchmark.robots import (
    DatasetGenerator,
    load_ai_concept_scores,
    load_ai_concepts,
    load_intervention_answers,
)
from concept_benchmark.robots.defaults import (
    AI_CONCEPT_SOURCES,
    CATALOG_SIZE,
    _load,
    image_index,
)

SEED = 1014


def _splits(preset: str):
    return DatasetGenerator(seed=SEED, concept_preset=preset).generate_splits()


@pytest.fixture(scope="module")
def datasets() -> dict:
    """Every supported way to hold the default robots."""
    discovered = _splits("foot_subtypes")
    oracle = _splits("ground_truth")
    noisy = _splits("foot_subtypes")
    noisy.sample_concept_noise(p=0.3, rng=0, enable=True)
    missing = _splits("foot_subtypes")
    missing.sample_concept_missingness(p=0.3, rng=0, enable=True)
    return {
        "discovered_full": discovered,
        "discovered_train": discovered.train,
        "discovered_test": discovered.test,
        "discovered_embedded": discovered.test.embed(
            torch.nn.Flatten(), batch_size=512
        ),
        "noisy_test": noisy.test,
        "missing_test": missing.test,
        "oracle_full": oracle,
        "oracle_test": oracle.test,
        "every_concept": DatasetGenerator(seed=SEED).generate(),
    }


@pytest.fixture(scope="module")
def clean_concepts(datasets) -> dict[str, np.ndarray]:
    """The true label of every concept for every catalog robot, from the full datasets."""
    truth = {}
    for name in ("every_concept",):
        data = datasets[name]
        rows = np.asarray(data.meta["df_indices"])
        for j, concept in enumerate(data.concepts):
            column = np.full(CATALOG_SIZE, -1)
            column[rows] = np.asarray(data.base_concepts)[:, j]
            truth[concept] = column
    return truth


def _rows(dataset) -> np.ndarray:
    return np.asarray(dataset.meta["df_indices"], dtype=int)


def _is_binary(a: np.ndarray) -> bool:
    return set(np.unique(a).tolist()) <= {0, 1}


# ---------------------------------------------------------------- rows


@pytest.mark.parametrize(
    "name",
    [
        "discovered_full",
        "discovered_train",
        "discovered_test",
        "noisy_test",
        "missing_test",
        "oracle_full",
        "oracle_test",
        "every_concept",
    ],
)
def test_rows_are_the_image_numbers(datasets, name):
    data = datasets[name]
    names = np.array([image_index(p) for p in data.inputs])
    assert np.array_equal(_rows(data), names)


def test_embedded_rows_are_the_rows_of_the_images(datasets):
    assert np.array_equal(
        _rows(datasets["discovered_embedded"]), _rows(datasets["discovered_test"])
    )


# ---------------------------------------------------------------- ai_concepts

SUPPORTED = [
    "discovered_full",
    "discovered_train",
    "discovered_test",
    "discovered_embedded",
    "noisy_test",
    "missing_test",
    "oracle_full",
    "oracle_test",
    "every_concept",
]


@pytest.mark.parametrize("source", sorted(AI_CONCEPT_SOURCES))
@pytest.mark.parametrize("name", SUPPORTED)
def test_ai_concepts_are_the_shipped_labels_of_each_robot(datasets, name, source):
    data = datasets[name]
    labels, names, texts = load_ai_concepts(data, source)
    shipped = _load("robot_ai_concepts.npz")[f"{AI_CONCEPT_SOURCES[source]}_labels"]
    assert labels.shape == (data.n, 12) and len(names) == len(texts) == 12
    assert _is_binary(labels)
    assert np.array_equal(labels, shipped[_rows(data)])


@pytest.mark.parametrize("source", sorted(AI_CONCEPT_SOURCES))
@pytest.mark.parametrize("name", SUPPORTED)
def test_ai_concept_scores_give_the_labels_at_one_half(datasets, name, source):
    data = datasets[name]
    scores, names, _ = load_ai_concept_scores(data, source)
    labels, label_names, _ = load_ai_concepts(data, source)
    shipped = _load("robot_ai_concepts.npz")[f"{AI_CONCEPT_SOURCES[source]}_scores"]
    assert scores.shape == labels.shape and names == label_names
    assert scores.min() >= 0 and scores.max() <= 1
    assert np.array_equal((scores >= 0.5).astype(int), labels)
    assert np.array_equal(scores, shipped[_rows(data)])


# ---------------------------------------------------------------- perfect


@pytest.mark.parametrize("name", SUPPORTED)
def test_perfect_answers_are_the_clean_labels(datasets, clean_concepts, name):
    data = datasets[name]
    answers = load_intervention_answers(data, "perfect")
    expected = np.stack([clean_concepts[c][_rows(data)] for c in data.concepts], axis=1)
    assert answers.shape == (data.n, len(data.concepts))
    assert _is_binary(answers)
    assert np.array_equal(answers, expected)


@pytest.mark.parametrize("name", ["discovered_test", "oracle_test", "every_concept"])
def test_perfect_answers_of_subsets_and_reorders(datasets, name):
    data = datasets[name]
    full = load_intervention_answers(data, "perfect")
    order = list(reversed(data.concepts))
    assert np.array_equal(
        load_intervention_answers(data, "perfect", concepts=order), full[:, ::-1]
    )
    concept = data.concepts[1]
    assert np.array_equal(
        load_intervention_answers(data, "perfect", concepts=[concept])[:, 0], full[:, 1]
    )


# ---------------------------------------------------------------- expert


@pytest.mark.parametrize("name", SUPPORTED)
def test_expert_answers_are_binary_and_seeded(datasets, name):
    data = datasets[name]
    answers = load_intervention_answers(data, "expert", seed=7)
    assert answers.shape == (data.n, len(data.concepts))
    assert _is_binary(answers)
    assert np.array_equal(answers, load_intervention_answers(data, "expert", seed=7))
    assert not np.array_equal(
        answers, load_intervention_answers(data, "expert", seed=8)
    )


def test_expert_is_right_at_the_given_rate(datasets, clean_concepts):
    data = datasets["every_concept"]
    answers = load_intervention_answers(data, "expert", accuracy=0.8, seed=7)
    perfect = load_intervention_answers(data, "perfect")
    assert abs((answers == perfect).mean() - 0.8) < 0.005


@pytest.mark.parametrize("accuracy", [0.0, 1.0])
def test_expert_at_the_extremes(datasets, accuracy):
    data = datasets["discovered_test"]
    answers = load_intervention_answers(data, "expert", accuracy=accuracy, seed=7)
    perfect = load_intervention_answers(data, "perfect")
    expected = perfect if accuracy == 1.0 else 1 - perfect
    assert np.array_equal(answers, expected)


def test_one_expert_answer_per_robot_and_concept(datasets):
    """Same seed: every dataset, split, subset and order gives the same answer per robot/concept."""
    seen: dict[str, np.ndarray] = {}
    for name in SUPPORTED:
        data = datasets[name]
        for order in (list(data.concepts), list(reversed(data.concepts))):
            answers = load_intervention_answers(data, "expert", concepts=order, seed=7)
            for j, concept in enumerate(order):
                column = seen.setdefault(concept, np.full(CATALOG_SIZE, -1))
                rows = _rows(data)
                known = column[rows] >= 0
                assert np.array_equal(column[rows][known], answers[known, j]), (
                    name,
                    concept,
                )
                column[rows] = answers[:, j]


# ---------------------------------------------------------------- llm


def _shipped_llm(questions: str, concepts: list[str], rows: np.ndarray) -> np.ndarray:
    data = _load("robot_llm_answers.npz")
    names = data[f"{questions}_concepts"].tolist()
    return data[f"{questions}_answers"][rows][:, [names.index(c) for c in concepts]]


LLM_CASES = [
    ("discovered_full", "human"),
    ("discovered_train", "human"),
    ("discovered_test", "human"),
    ("discovered_embedded", "human"),
    ("noisy_test", "human"),
    ("missing_test", "human"),
    ("oracle_full", "true"),
    ("oracle_test", "true"),
]


@pytest.mark.parametrize("name,questions", LLM_CASES)
def test_llm_answers_come_from_the_dataset_question_list(datasets, name, questions):
    data = datasets[name]
    answers = load_intervention_answers(data, "llm")
    assert _is_binary(answers)
    assert np.array_equal(
        answers, _shipped_llm(questions, list(data.concepts), _rows(data))
    )


@pytest.mark.parametrize("name,questions", LLM_CASES)
def test_llm_answers_of_subsets_and_reorders(datasets, name, questions):
    data = datasets[name]
    full = load_intervention_answers(data, "llm")
    order = list(reversed(data.concepts))
    assert np.array_equal(
        load_intervention_answers(data, "llm", concepts=order), full[:, ::-1]
    )
    for j, concept in enumerate(data.concepts):
        alone = load_intervention_answers(data, "llm", concepts=[concept])[:, 0]
        assert np.array_equal(alone, full[:, j]), concept


@pytest.mark.parametrize("source", sorted(AI_CONCEPT_SOURCES))
@pytest.mark.parametrize("name", SUPPORTED)
def test_llm_answers_on_automated_concepts(datasets, name, source):
    data = datasets[name]
    _, names, _ = load_ai_concepts(data, source)
    questions = {
        "machine_labeled": "human",
        "llm_written": "llm",
        "clip_words": "clip",
    }[source]
    answers = load_intervention_answers(data, "llm", concepts=names)
    assert np.array_equal(answers, _shipped_llm(questions, names, _rows(data)))


def test_llm_answers_on_every_concept_dataset(datasets):
    data = datasets["every_concept"]
    oracle_names = _load("robot_llm_answers.npz")["true_concepts"].tolist()
    answers = load_intervention_answers(data, "llm", concepts=oracle_names)
    assert np.array_equal(answers, _shipped_llm("true", oracle_names, _rows(data)))
    with pytest.raises(ValueError):
        load_intervention_answers(data, "llm")  # has_elbows, hand_shape: never asked


# ---------------------------------------------------------------- unsupported


@pytest.fixture(scope="module")
def unsupported() -> dict:
    return {
        "text": DatasetGenerator(seed=SEED, data_type="text").generate(),
        "other_catalog": DatasetGenerator(
            seed=SEED, renders_per_robot=2, render_images=False
        ).generate(),
    }


@pytest.mark.parametrize("name", ["text", "other_catalog"])
@pytest.mark.parametrize(
    "call",
    [
        lambda d: load_ai_concepts(d, "llm_written"),
        lambda d: load_intervention_answers(d, "perfect"),
        lambda d: load_intervention_answers(d, "expert"),
        lambda d: load_intervention_answers(d, "llm"),
    ],
)
def test_other_datasets_are_rejected_clearly(unsupported, name, call):
    with pytest.raises(ValueError):
        call(unsupported[name])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "oracle"},
        {"source": "expert", "accuracy": 80},
        {"source": "expert", "accuracy": -0.1},
        {"source": "expert", "seed": -1},
        {"source": "expert", "seed": 1.5},
        {"source": "perfect", "concepts": ["Solid colored"]},
        {"source": "llm", "concepts": ["not_a_concept"]},
    ],
)
def test_bad_arguments_are_rejected_clearly(datasets, kwargs):
    with pytest.raises(ValueError):
        load_intervention_answers(datasets["discovered_test"], **kwargs)


def test_partial_metadata_is_rejected_clearly(datasets):
    data = copy.deepcopy(datasets["discovered_full"])
    del data.meta["unfiltered_concepts"]
    with pytest.raises(ValueError):
        load_intervention_answers(data, "perfect")


def test_unknown_ai_source_is_rejected(datasets):
    with pytest.raises(ValueError):
        load_ai_concepts(datasets["discovered_test"], "human")
