"""Build the default annotations and intervention answers shipped in the pip package.

Writes two files to `concept_benchmark/data/defaults/`:

- `robot_llm_answers.npz`: the answer of Gemini 2.5 Flash-Lite (224 px images) for every robot image
  and concept, for the four question lists `true`, `human`, `llm`, `clip`. Each answer is the first
  one the paper's caches hold for that image, seeds 1014 to 1023 in order, then the images that were
  never in a test set (`missing-tests` lists them; `build_llm_caches.py` asks about them).
- `robot_ai_concepts.npz`: concept scores and labels (score >= 0.5) of the three automated concept sets `machine`,
  `llm`, `clip` for every robot image, from the seed-1014 label-free models of the paper's grid
  applied to CLIP embeddings of the 32 px images.

    python scripts/paper/build_pip_defaults.py missing-tests --out results/_incoming/llm_caches_missing/missing_tests.json
    python scripts/paper/build_pip_defaults.py llm-answers --missing-dir results/_incoming/llm_caches_missing
    python scripts/paper/build_pip_defaults.py ai-concepts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import REPO
from build_llm_caches import build_concept_lists, sha1_of
from concept_benchmark.ext.fileutils import load
from concept_benchmark.robots.defaults import (
    AI_CONCEPT_SOURCES,
    CATALOG_SIZE,
    LLM_QUESTION_LISTS,
    image_index,
)
from experiments.lfcbm import _CLIPEncoder

SEEDS = [str(s) for s in range(1014, 1024)]
PAPER_CACHE_TAGS = {
    "true": "true",
    "human": "human-and-machine",
    "llm": "llm",
    "clip": "clip",
}
PAPER_CACHES = REPO / "results/paper/robot/llm_caches"
NEW_SEED_CACHES = REPO / "results/_incoming/llm_caches_n10"
SEED_TESTS = NEW_SEED_CACHES / "seed_tests.json"
OUT_DIR = REPO / "concept_benchmark/data/defaults"
LFCBM_MODELS = REPO / "results/paper/models/robot/grid"


def read_cache(path: Path, n_concepts: int) -> dict[int, list[int]]:
    """Answers of a cache by test row; every row must answer all ``n_concepts`` concepts."""
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        votes = row["votes_idx"]
        if sorted(votes, key=int) != [str(j) for j in range(n_concepts)]:
            raise ValueError(
                f"{path.name}: row {row['i']} does not answer every concept"
            )
        rows[int(row["i"])] = [int(votes[str(j)]) for j in range(n_concepts)]
    return rows


def cache_path(seed: str, questions: str, names: list[str], dsig: str) -> Path:
    tagged = (
        PAPER_CACHES
        / f"robot__concepts-{PAPER_CACHE_TAGS[questions]}__llm-gemini-2.5-flash-lite__img-224px"
        f"__questions-v2-value-explicit__seed-{seed}__llm-votes.jsonl"
    )
    if tagged.exists():
        return tagged
    return NEW_SEED_CACHES / f"llm_interventions_{sha1_of(names)}_{dsig}.jsonl"


def write_missing_tests(out: Path) -> None:
    tests = json.loads(SEED_TESTS.read_text())
    seen = {image_index(p) for test in tests.values() for p in test["inputs"]}
    inputs = [f"robot_{i:03d}.png" for i in range(CATALOG_SIZE) if i not in seen]
    first = tests[SEEDS[0]]
    entry = {
        "inputs": inputs,
        "dsig": sha1_of(inputs),
        "true_names": first["true_names"],
        "human_names": first["human_names"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"missing": entry}))
    print(f"{len(inputs)} images never in a test set -> {out}")


def build_llm_answers(missing_dir: Path | None) -> None:
    tests = json.loads(SEED_TESTS.read_text())
    if missing_dir is not None:
        tests["missing"] = json.loads((missing_dir / "missing_tests.json").read_text())[
            "missing"
        ]
    first = tests[SEEDS[0]]
    lists = build_concept_lists(first["true_names"], first["human_names"])
    arrays = {}
    for questions in LLM_QUESTION_LISTS:
        names = [n for n, _ in lists[questions]]
        answers = np.full((CATALOG_SIZE, len(names)), -1, dtype=np.int8)
        for seed in SEEDS + (["missing"] if missing_dir is not None else []):
            test = tests[seed]
            path = (
                missing_dir / f"llm_interventions_{sha1_of(names)}_{test['dsig']}.jsonl"
                if seed == "missing"
                else cache_path(seed, questions, names, test["dsig"])
            )
            for row, votes in read_cache(path, len(names)).items():
                image = image_index(test["inputs"][row])
                if answers[image, 0] < 0:
                    answers[image] = votes
        unanswered = int((answers < 0).any(axis=1).sum())
        if unanswered:
            raise ValueError(
                f"{questions}: {unanswered} images without answers; ask the LLM about them "
                "(missing-tests, then build_llm_caches.py) and pass --missing-dir"
            )
        arrays[f"{questions}_answers"] = answers
        arrays[f"{questions}_concepts"] = np.array(names)
        print(f"{questions:5}: {len(names)} concepts, every image answered")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "robot_llm_answers.npz", **arrays)
    print(f"-> {OUT_DIR / 'robot_llm_answers.npz'}")


def build_ai_concepts(device: str) -> None:
    images = [
        REPO / f"data/robot_images/robot_{i:03d}.png" for i in range(CATALOG_SIZE)
    ]
    arrays = {}
    for source in AI_CONCEPT_SOURCES.values():
        (path,) = LFCBM_MODELS.glob(
            f"robot__rule-sparse__concepts-{source}__arch-lfcbm__*seed-1014__model.pt"
        )
        model = load(path)
        model.cfg.cache_dir = None
        # set the encoder here: the model would otherwise pick the best available device
        model.encoder = _CLIPEncoder(
            model.cfg.clip_model, model.cfg.clip_pretrained, device
        )
        proba = model.concept_proba(images).astype(np.float32)
        arrays[f"{source}_scores"] = proba
        arrays[f"{source}_labels"] = (proba >= 0.5).astype(np.uint8)
        arrays[f"{source}_concepts"] = np.array(list(model.concept_set.keys))
        arrays[f"{source}_texts"] = np.array(list(model.concept_set.texts))
        print(
            f"{source:7}: {proba.shape}, positive share {arrays[f'{source}_labels'].mean(0).round(2)}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "robot_ai_concepts.npz", **arrays)
    print(f"-> {OUT_DIR / 'robot_ai_concepts.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="cmd", required=True)
    missing = commands.add_parser("missing-tests")
    missing.add_argument("--out", type=Path, required=True)
    answers = commands.add_parser("llm-answers")
    answers.add_argument("--missing-dir", type=Path, default=None)
    concepts = commands.add_parser("ai-concepts")
    concepts.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.cmd == "missing-tests":
        write_missing_tests(args.out)
    elif args.cmd == "llm-answers":
        build_llm_answers(args.missing_dir)
    else:
        build_ai_concepts(args.device)


if __name__ == "__main__":
    main()
