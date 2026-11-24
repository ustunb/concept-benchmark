"""Build the default annotations and intervention answers shipped in the pip package.

Writes two files to `concept_benchmark/data/defaults/`:

- `robot_llm_answers.npz`: the answer of Gemini 2.5 Flash-Lite (224 px images) for every robot image
  and concept, for the four question lists `true`, `human`, `llm`, `clip`. Each answer is the first
  one the paper's caches hold for that image, seeds 1014 to 1023 in order, then the images that were
  never in a test set (`missing-tests` lists them; `build_llm_caches.py` asks about them).
- `robot_ai_concepts.npz`: concept probabilities and labels (p >= 0.5) of the three automated concept
  sets `machine`, `llm`, `clip` for every robot image, from the seed-1014 label-free models of the
  paper's grid applied to CLIP embeddings of the 32 px images.

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

N_IMAGES = 30720
SEEDS = [str(s) for s in range(1014, 1024)]
QUESTION_LISTS = ["true", "human", "llm", "clip"]
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
AI_SOURCES = ["machine", "llm", "clip"]


def image_index(name: str) -> int:
    return int(name.split("_")[1].split(".")[0])


def read_cache(path: Path) -> dict[int, list[int]]:
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        votes = row["votes_idx"]
        rows[int(row["i"])] = [int(votes[str(j)]) for j in range(len(votes))]
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


def missing_tests(out: Path) -> None:
    tests = json.loads(SEED_TESTS.read_text())
    seen = {image_index(p) for t in tests.values() for p in t["inputs"]}
    inputs = [f"robot_{i:03d}.png" for i in range(N_IMAGES) if i not in seen]
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


def llm_answers(missing_dir: Path | None) -> None:
    tests = json.loads(SEED_TESTS.read_text())
    if missing_dir is not None:
        tests["missing"] = json.loads((missing_dir / "missing_tests.json").read_text())[
            "missing"
        ]
    first = tests[SEEDS[0]]
    lists = build_concept_lists(first["true_names"], first["human_names"])
    arrays = {}
    for questions in QUESTION_LISTS:
        names = [n for n, _ in lists[questions]]
        answers = np.full((N_IMAGES, len(names)), -1, dtype=np.int8)
        for seed in SEEDS + (["missing"] if missing_dir is not None else []):
            t = tests[seed]
            path = (
                missing_dir / f"llm_interventions_{sha1_of(names)}_{t['dsig']}.jsonl"
                if seed == "missing"
                else cache_path(seed, questions, names, t["dsig"])
            )
            rows = read_cache(path)
            for i, votes in rows.items():
                k = image_index(t["inputs"][i])
                if answers[k, 0] < 0:
                    answers[k] = votes
        arrays[f"{questions}_answers"] = answers
        arrays[f"{questions}_concepts"] = np.array(names)
        print(
            f"{questions:5}: {len(names)} concepts, {int((answers[:, 0] < 0).sum())} images without answers"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "robot_llm_answers.npz", **arrays)
    print(f"-> {OUT_DIR / 'robot_llm_answers.npz'}")


def ai_concepts(device: str) -> None:
    from concept_benchmark.ext.fileutils import load

    images = [REPO / f"data/robot_images/robot_{i:03d}.png" for i in range(N_IMAGES)]
    arrays = {}
    for source in AI_SOURCES:
        (path,) = LFCBM_MODELS.glob(
            f"robot__rule-sparse__concepts-{source}__arch-lfcbm__*seed-1014__model.pt"
        )
        model = load(path)
        model.cfg.cache_dir = None
        model.cfg.device = device
        if hasattr(model, "_encoder") and model._encoder is not None:
            model._encoder.device = device
        proba = model.concept_proba(images)
        arrays[f"{source}_proba"] = proba.astype(np.float16)
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
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("missing-tests")
    m.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("llm-answers")
    a.add_argument("--missing-dir", type=Path, default=None)
    c = sub.add_parser("ai-concepts")
    c.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.cmd == "missing-tests":
        missing_tests(args.out)
    elif args.cmd == "llm-answers":
        llm_answers(args.missing_dir)
    else:
        ai_concepts(args.device)


if __name__ == "__main__":
    main()
