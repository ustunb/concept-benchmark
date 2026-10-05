"""Fill new seeds' LLM intervention caches with answers Gemini already gave for the same image and questions.

An answer depends only on the image and the question list, and the test robots of different seeds overlap. For each
concept list, the answer per image is taken from the installed caches of seeds 1014-1017 (Gemini 2.5 Flash-Lite,
224 px, value-explicit questions; first seed in that order wins) and, for images none of them covers, from rows
already written to the target files. The target caches are rewritten in the builder's format; afterwards
scripts/paper/build_llm_caches.py asks Gemini only about the robots still missing (it resumes from these files).

    python scripts/paper/assemble_llm_caches.py --seed-tests results/_incoming/llm_caches_n10/seed_tests.json \
        --out-dir results/_incoming/llm_caches_n10 --seeds 1018 1019 1020 1021 1022 1023
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _common import PAPER_RESULTS
from build_llm_caches import build_concept_lists

INSTALLED = PAPER_RESULTS / "robot/llm_caches"
SOURCE_SEEDS = ("1014", "1015", "1016", "1017")
LIST_TAG = {"true": "true", "human": "human-and-machine", "llm": "llm", "clip": "clip"}
VARIANT = "llm-gemini-2.5-flash-lite__img-224px__questions-v2-value-explicit"


def sha1_names(names: list[str]) -> str:
    h = hashlib.sha1()
    for n in names:
        h.update(str(n).encode())
        h.update(b"\x00")
    return h.hexdigest()


def read_rows(path: Path, n: int) -> dict[int, list[int]]:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if len(r["votes_idx"]) == n:
                rows[int(r["i"])] = [r["votes_idx"][str(j)] for j in range(n)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-tests", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    args = ap.parse_args()
    tests = json.loads(args.seed_tests.read_text())
    for cache, tag in LIST_TAG.items():
        answers: dict[str, list[int]] = {}
        disagree = overlap = 0
        for s in SOURCE_SEEDS:
            t = tests[s]
            n = len(build_concept_lists(t["true_names"], t["human_names"])[cache])
            rows = read_rows(INSTALLED / f"robot__concepts-{tag}__{VARIANT}__seed-{s}__llm-votes.jsonl", n)
            if len(rows) != len(t["inputs"]):
                raise SystemExit(f"{cache} seed {s}: installed cache has {len(rows)} rows, expected {len(t['inputs'])}")
            for i, votes in rows.items():
                img = t["inputs"][i]
                if img in answers:
                    overlap += 1
                    disagree += answers[img] != votes
                else:
                    answers[img] = votes
        print(f"{cache:5}: {len(answers)} images answered in seeds 1014-1017; same image answered twice: {overlap}, "
              f"of which differently: {disagree}", flush=True)
        for s in args.seeds:
            t = tests[s]
            concepts = build_concept_lists(t["true_names"], t["human_names"])[cache]
            path = args.out_dir / f"llm_interventions_{sha1_names([c for c, _ in concepts])}_{t['dsig']}.jsonl"
            fresh = read_rows(path, len(concepts))  # rows from the stopped build, used only where nothing else exists
            out, reused, kept = [], 0, 0
            for i, img in enumerate(t["inputs"]):
                if img in answers:
                    out.append((i, answers[img])); reused += 1
                elif i in fresh:
                    out.append((i, fresh[i])); kept += 1
            with path.open("w") as fh:
                for i, votes in out:
                    fh.write(json.dumps({"i": i, "votes_idx": {str(j): v for j, v in enumerate(votes)}}) + "\n")
            print(f"  seed {s}: reused {reused}, kept {kept} new answers, still to ask {len(t['inputs']) - len(out)}",
                  flush=True)


if __name__ == "__main__":
    main()
