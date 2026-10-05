"""Build exhaustive LLM intervention caches for the robots benchmark.

Asks a Gemini model every concept for every test robot of each seed and writes one cache per
(seed, concept list) in the format `robot_pipeline.py --llm-cache-only` reads:
`llm_interventions_{sha1(concept names)}_{sha1(test inputs)}.jsonl`, rows `{"i", "votes_idx"}`.

Images are labelled inline (`img_k:` before each image) and answers are keyed by label, so an
extra, missing or malformed answer can never shift onto another robot; such robots are re-asked.
The run is resumable: robots already complete in a cache file are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from _common import REPO

BATCH_SIZE = 10
MAX_ATTEMPTS = 8
PRICE_IN, PRICE_OUT = 0.10, 0.40  # USD per 1M tokens, gemini-2.5-flash-lite paid tier

TRUE_QUESTIONS = {
    "head_shape": "1 if the head is ROUND, 0 if the head is SQUARE",
    "body_shape": "1 if the body is ROUND, 0 if the body is SQUARE",
    "has_knees": "1 if the legs have visible KNEE joints, 0 if not",
    "has_antennae": "1 if the robot has ANTENNAE on its head, 0 if not",
    "ears_shape": "1 if the ears are TRIANGULAR, 0 if the ears are SQUARE",
    "mouth_type": "1 if the mouth is OPEN, 0 if the mouth is CLOSED",
    "foot_shape": "1 if the feet are POINTY, 0 if the feet are FLAT",
}


def sha1_of(items) -> str:
    h = hashlib.sha1()
    for x in items:
        h.update(str(x).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def read_descriptions(name: str) -> list[dict]:
    raw = subprocess.check_output(
        ["git", "show", f"origin/balanced-baseline:concept_benchmark/concept_descriptions/{name}.jsonl"],
        cwd=REPO,
    ).decode()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def build_concept_lists(true_names: list[str], human_names: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Return {cache: [(concept_name, question)]}, names in the order the pipeline hashes them."""
    subtypes = {r["key"]: r["text"] for r in read_descriptions("gt_concepts_subconcept")}

    def human_question(name: str) -> str:
        if name in TRUE_QUESTIONS:
            return TRUE_QUESTIONS[name]
        return f"1 if the robot has {subtypes[name].replace('a robot with ', '')}, 0 otherwise"

    return {
        "true": [(n, TRUE_QUESTIONS[n]) for n in true_names],
        "human": [(n, human_question(n)) for n in human_names],
        "llm": [
            (r.get("key", r["text"]), f"1 if this describes the robot: '{r['text']}', 0 otherwise")
            for r in read_descriptions("llm")
        ],
        "clip": [
            (
                r.get("key", r["text"]),
                f"1 if the word '{r['text']}' applies to the robot or to something visible in the image, 0 otherwise",
            )
            for r in read_descriptions("clip")
        ],
    }


class CacheFile:
    """Append-only cache with a lock; knows which robots are already complete."""

    def __init__(self, path: Path, n_concepts: int):
        self.path, self.n = path, n_concepts
        self.lock = threading.Lock()
        self.done: set[int] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if len(row["votes_idx"]) == n_concepts:
                    self.done.add(int(row["i"]))

    def append(self, rows: list[tuple[int, list[int]]]) -> None:
        with self.lock, self.path.open("a") as fh:
            for i, votes in rows:
                if i in self.done:
                    continue
                fh.write(json.dumps({"i": i, "votes_idx": {str(j): v for j, v in enumerate(votes)}}) + "\n")
                self.done.add(i)


class Budget:
    def __init__(self, max_usd: float):
        self.max_usd, self.usd, self.lock = max_usd, 0.0, threading.Lock()

    def add(self, tokens_in: int, tokens_out: int) -> None:
        with self.lock:
            self.usd += (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6

    def exhausted(self) -> bool:
        return self.usd >= self.max_usd


def ask(client, model: str, images: list[Path], concepts: list[tuple[str, str]], budget: Budget):
    """One labelled request; returns {label_index: votes} for complete, well-formed answers only."""
    labels = [f"img_{k}" for k in range(len(images))]
    keys = [f"q{j}" for j in range(len(concepts))]
    rules = "\n".join(f'- "{k}": {q}' for k, (_, q) in zip(keys, concepts))
    parts = [
        "Each robot image below is preceded by its label. For EVERY label answer all questions with 0 or 1:\n"
        f"{rules}\n\nReturn ONLY a JSON object whose keys are exactly these labels: {labels}. "
        f"Each value is an object with exactly the keys {keys}."
    ]
    # enforce the labelled object: the model otherwise sometimes returns an ordered array,
    # which cannot be tied to images and is exactly the format that misaligned answers before
    item = types.Schema(
        type=types.Type.OBJECT,
        properties={k: types.Schema(type=types.Type.INTEGER) for k in keys},
        required=keys,
        property_ordering=keys,
    )
    schema = types.Schema(
        type=types.Type.OBJECT, properties={lab: item for lab in labels}, required=labels, property_ordering=labels
    )
    for label, path in zip(labels, images):
        parts += [f"{label}:", Image.open(path).convert("RGB")]
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema),
            )
            u = resp.usage_metadata
            budget.add(u.prompt_token_count or 0, u.candidates_token_count or 0)
            obj = json.loads(resp.text)
            break
        except json.JSONDecodeError:
            return {}
        except Exception as e:  # rate limits and transient server errors: back off and retry
            msg = str(e)
            if not any(code in msg for code in ("429", "500", "502", "503", "504")):
                raise
            wait = min(120, 5 * 2**attempt) + random.random() * 5
            print(f"  retry {attempt + 1}: {msg[:70]} (sleep {wait:.0f}s)", flush=True)
            time.sleep(wait)
    else:
        return {}
    out = {}
    for k, label in enumerate(labels):
        ans = obj.get(label)
        if isinstance(ans, dict) and all(key in ans for key in keys):
            try:
                out[k] = [1 if int(ans[key]) else 0 for key in keys]
            except (TypeError, ValueError):
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-tests", type=Path, required=True, help="JSON with per-seed test inputs and concept names")
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--caches", nargs="+", default=["true", "human", "llm", "clip"])
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N test robots per seed (dry runs)")
    args = ap.parse_args()

    tests = json.loads(args.seed_tests.read_text())
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    budget = Budget(args.max_usd)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in args.seeds:
        t = tests[seed]
        inputs = t["inputs"][: args.limit or None]
        lists = build_concept_lists(t["true_names"], t["human_names"])
        for cache in args.caches:
            concepts = lists[cache]
            name = f"llm_interventions_{sha1_of([n for n, _ in concepts])}_{t['dsig']}.jsonl"
            cf = CacheFile(args.out_dir / name, len(concepts))
            todo = [i for i in range(len(inputs)) if i not in cf.done]
            print(f"seed {seed} {cache:5} -> {name[:40]}… done {len(cf.done)}, todo {len(todo)}", flush=True)
            for s in range(0, len(todo), BATCH_SIZE):
                jobs.append((seed, cache, cf, concepts, todo[s : s + BATCH_SIZE], inputs))

    def run(job):
        seed, cache, cf, concepts, idxs, inputs = job
        pending = list(idxs)
        for _ in range(4):  # re-ask robots whose answers came back missing or malformed
            if not pending or budget.exhausted():
                break
            got = ask(client, args.model, [args.image_dir / inputs[i] for i in pending], concepts, budget)
            cf.append([(pending[k], v) for k, v in got.items()])
            pending = [i for k, i in enumerate(pending) if k not in got]
        return len(idxs) - len(pending), pending

    done = failed = 0
    started = time.time()
    caches = {id(j[2]): j[2] for j in jobs}

    def report():
        while True:
            time.sleep(30)
            rows = sum(len(c.done) for c in caches.values())
            print(f"[{time.time() - started:.0f}s] rows written {rows} | ${budget.usd:.3f}", flush=True)

    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            ok, pending = fut.result()
            done += ok
            failed += len(pending)
            if n % 50 == 0 or n == len(futures):
                print(
                    f"{n}/{len(futures)} batches | robots answered {done}, unanswered {failed} | "
                    f"${budget.usd:.2f} | {time.time() - started:.0f}s",
                    flush=True,
                )
            if budget.exhausted():
                print(f"STOP: budget ${args.max_usd} reached; rerun to resume", flush=True)
                pool.shutdown(cancel_futures=True)
                break


if __name__ == "__main__":
    main()
