"""Export each seed's test robots and concept names for scripts/paper/build_llm_caches.py (balanced rule).

Reads the installed balanced-rule datasets (true and human concepts share the same test robots) and writes the
`--seed-tests` JSON: per seed the test image names, their signature (the cache key's second half), the concept names
and the concept values.

    python scripts/paper/export_seed_tests.py --seeds 1018 1019 --out results/_incoming/llm_caches_n10/seed_tests.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from concept_benchmark.ext.fileutils import load  # noqa: E402

DATASETS = REPO / "results/paper/robot/datasets"
TAG = "rule-balanced__sampling-skew0.30__elbows-weight2"


def signature(items) -> str:
    h = hashlib.sha1()
    for x in items:
        h.update(str(x).encode())
        h.update(b"\x00")
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for s in args.seeds:
        true = load(DATASETS / f"robot__{TAG}__concepts-true__seed-{s}__dataset.data").test
        human = load(DATASETS / f"robot__{TAG}__concepts-human__seed-{s}__dataset.data").test
        if list(map(str, true.inputs)) != list(map(str, human.inputs)):
            raise SystemExit(f"seed {s}: true and human datasets differ in test robots")
        out[str(s)] = {"inputs": [str(x) for x in true.inputs], "dsig": signature(true.inputs),
                       "true_names": list(true.concepts), "human_names": list(human.concepts),
                       "true_C": [list(map(int, r)) for r in true.C], "human_C": [list(map(int, r)) for r in human.C]}
        print(s, "dsig", out[str(s)]["dsig"][:12], "robots", len(true.inputs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
