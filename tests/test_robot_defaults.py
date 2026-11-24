"""The default automated concept labels and intervention answers shipped with the package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from concept_benchmark.robots import DatasetGenerator, ai_concepts, intervention_answers
from concept_benchmark.robots.defaults import CATALOG_SIZE, _load, image_index

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def discovered():
    return DatasetGenerator(seed=1014, concept_preset="foot_subtypes").generate_splits()


@pytest.mark.parametrize("questions", ["true", "human", "llm", "clip"])
def test_every_robot_image_has_llm_answers(questions):
    answers = _load("robot_llm_answers.npz")[f"{questions}_answers"]
    assert answers.shape[0] == CATALOG_SIZE
    assert set(np.unique(answers)) <= {0, 1}


@pytest.mark.parametrize("source", ["machine_labeled", "llm_written", "clip_words"])
def test_ai_concepts_label_every_robot(discovered, source):
    labels, names, texts = ai_concepts(discovered.test, source)
    assert labels.shape == (discovered.test.n, len(names))
    assert len(names) == len(texts) == 12
    assert set(np.unique(labels)) <= {0, 1}


def test_perfect_answers_are_the_true_labels(discovered):
    test = discovered.test
    assert np.array_equal(intervention_answers(test, "perfect"), np.asarray(test.C))


def test_perfect_answers_follow_the_concept_order(discovered):
    test = discovered.test
    reordered = intervention_answers(
        test, "perfect", concepts=list(reversed(test.concepts))
    )
    assert np.array_equal(reordered, np.asarray(test.C)[:, ::-1])


def test_perfect_answers_need_the_dataset_concepts(discovered):
    with pytest.raises(ValueError):
        intervention_answers(discovered.test, "perfect", concepts=["not_a_concept"])


def test_perfect_answers_on_the_full_dataset(discovered):
    assert np.array_equal(
        intervention_answers(discovered, "perfect"),
        np.asarray(discovered.base_concepts),
    )


def test_perfect_answers_ignore_concept_noise():
    data = DatasetGenerator(seed=1014, concept_preset="foot_subtypes").generate_splits()
    clean = np.asarray(data.test.C).copy()
    data.sample_concept_noise(p=0.3, rng=0, enable=True)
    assert not np.array_equal(np.asarray(data.test.C), clean)
    assert np.array_equal(intervention_answers(data.test, "perfect"), clean)


def test_answers_follow_the_rows_not_the_file_names(discovered):
    test = discovered.test
    rows = np.asarray(test.meta["df_indices"])
    names = np.array([image_index(p) for p in test.inputs])
    assert np.array_equal(rows, names)


def test_expert_answers_are_right_at_the_given_rate(discovered):
    test = discovered.test
    answers = intervention_answers(test, "expert", accuracy=0.8, seed=3)
    assert abs((answers == np.asarray(test.C)).mean() - 0.8) < 0.01
    assert np.array_equal(
        answers, intervention_answers(test, "expert", accuracy=0.8, seed=3)
    )


def test_expert_answers_do_not_depend_on_the_other_concepts(discovered):
    test = discovered.test
    full = intervention_answers(test, "expert", seed=5)
    k = list(test.concepts).index("ears_shape")
    alone = intervention_answers(test, "expert", concepts=["ears_shape"], seed=5)
    assert np.array_equal(alone[:, 0], full[:, k])


def test_expert_accuracy_must_be_a_fraction(discovered):
    with pytest.raises(ValueError):
        intervention_answers(discovered.test, "expert", accuracy=80)


def test_another_catalog_is_rejected():
    other = DatasetGenerator(
        seed=1014, renders_per_robot=2, render_images=False
    ).generate()
    with pytest.raises(ValueError):
        intervention_answers(other, "perfect")


def test_llm_answers_follow_the_concept_order(discovered):
    test = discovered.test
    answers = intervention_answers(test, "llm")
    assert answers.shape == np.asarray(test.C).shape
    reordered = intervention_answers(
        test, "llm", concepts=list(reversed(test.concepts))
    )
    assert np.array_equal(reordered, answers[:, ::-1])


def test_llm_answers_of_a_subset_come_from_the_same_questions(discovered):
    test = discovered.test
    full = intervention_answers(test, "llm")
    k = list(test.concepts).index("ears_shape")
    assert np.array_equal(
        intervention_answers(test, "llm", concepts=["ears_shape"])[:, 0], full[:, k]
    )


def test_llm_answers_for_a_dataset_with_every_concept():
    full = DatasetGenerator(seed=1014).generate()
    answers = intervention_answers(full, "llm", concepts=["head_shape", "foot_shape"])
    assert answers.shape == (full.n, 2)


def test_llm_answers_on_ai_concepts(discovered):
    _, names, _ = ai_concepts(discovered.test, "llm_written")
    answers = intervention_answers(discovered.test, "llm", concepts=names)
    assert answers.shape == (discovered.test.n, 12)


def test_unknown_concepts_have_no_llm_answers(discovered):
    with pytest.raises(ValueError):
        intervention_answers(discovered.test, "llm", concepts=["not_a_concept"])


PAPER_CACHE = (
    REPO
    / "results/paper/robot/llm_caches"
    / "robot__concepts-human-and-machine__llm-gemini-2.5-flash-lite__img-224px"
    "__questions-v2-value-explicit__seed-1014__llm-votes.jsonl"
)
SEED_TESTS = REPO / "results/_incoming/llm_caches_n10/seed_tests.json"


@pytest.mark.skipif(
    not (PAPER_CACHE.exists() and SEED_TESTS.exists()),
    reason="needs the paper's LLM caches",
)
def test_llm_answers_match_the_paper_on_seed_1014():
    inputs = json.loads(SEED_TESTS.read_text())["1014"]["inputs"]
    data = _load("robot_llm_answers.npz")
    for line in PAPER_CACHE.read_text().splitlines():
        row = json.loads(line)
        k = image_index(inputs[row["i"]])
        votes = [row["votes_idx"][str(j)] for j in range(len(row["votes_idx"]))]
        assert data["human_answers"][k].tolist() == votes
