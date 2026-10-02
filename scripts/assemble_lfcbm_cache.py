"""Build a run directory's label-free CBM caches from CLIP embeddings computed for other datasets.

CLIP image embeddings depend only on the image, but the caches store them per split: the label-free CBM reads
`lfcbm_<regime>_cache/clip_img_{train,valid}.npy` by name (experiments/lfcbm.py `fit`) and the shared cache by a
hash of the split's path list (`transform`). This collects one embedding per image from the grid runs' caches
(rows follow each grid dataset's split order), embeds the images no cache covers with the same CLIP model, and
writes the target run's caches in its own split order; the concept text embeddings are copied per regime.
A positive control re-embeds images that are cached and requires cosine similarity >= 0.999.

    cd <target run dir> && PYTHONPATH=. python assemble_lfcbm_cache.py --seed 1014 \
        --dataset results/robot_image_4_subconcept_seed1014.data \
        --grid-run /ocean/.../run_s1014 --grid-run /ocean/.../run_s1015 ... --grid-old-root-pattern /home/jskirzynski/cb-run-s{seed}
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np

from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import data_dir
from experiments.lfcbm import LFTrainingConfig, _CLIPEncoder

REGIMES = ("machine", "llm", "clip")
SPLITS = ("train", "validation", "test")


def path_hash_name(paths: list[str]) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(str(p).encode("utf-8"))
    h.update(str(len(paths)).encode("utf-8"))
    return f"clip_img_{h.hexdigest()[:16]}.npy"


def collect(grid_run: Path, old_root: str) -> dict[str, np.ndarray]:
    """file name -> embedding, from one grid run's per-regime and shared caches."""
    seed = re.search(r"run_s(\d+)", str(grid_run)).group(1)
    res = grid_run / "results"
    data = load(res / f"robot_image_4_subconcept_seed{seed}.data")
    lf = res / f"lfcbm_seed{seed}"
    found: dict[str, np.ndarray] = {}
    regime_cache = lf / "lfcbm_machine_cache"
    for split, fname in (("train", "clip_img_train.npy"), ("validation", "clip_img_valid.npy")):
        arr = np.load(regime_cache / fname)
        files = list(map(str, getattr(data, split).inputs))
        assert len(arr) == len(files), (grid_run, split, arr.shape, len(files))
        found.update(zip(files, arr))
    for split in SPLITS:
        files = list(map(str, getattr(data, split).inputs))
        shared = lf / "lfcbm_shared_img_cache" / path_hash_name([f"{old_root}/data/robot_images/{p}" for p in files])
        if shared.exists():
            found.update(zip(files, np.load(shared)))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="Target run's subconcept dataset.")
    ap.add_argument("--grid-run", type=Path, action="append", required=True)
    ap.add_argument("--grid-old-root-pattern", required=True, help="Repo folder the grid LFCBMs ran in, with {seed}.")
    ap.add_argument("--control", type=int, default=200)
    args = ap.parse_args()

    pool: dict[str, np.ndarray] = {}
    for run in args.grid_run:
        seed = re.search(r"run_s(\d+)", str(run)).group(1)
        pool.update(collect(run, args.grid_old_root_pattern.format(seed=seed)))
    print(f"embeddings collected for {len(pool)} images", flush=True)

    data = load(args.dataset)
    image_dir = data_dir / "robot_images"
    splits = {s: list(map(str, getattr(data, s).inputs)) for s in SPLITS}
    need = sorted({p for files in splits.values() for p in files if p not in pool})
    cfg = LFTrainingConfig()
    encoder = _CLIPEncoder(cfg.clip_model, cfg.clip_pretrained, cfg.device)
    rng = np.random.default_rng(0)
    cached = sorted({p for files in splits.values() for p in files if p in pool})
    control = [cached[i] for i in rng.choice(len(cached), size=min(args.control, len(cached)), replace=False)]
    fresh = encoder.encode_images([str(image_dir / p) for p in control], cfg.batch_size)
    cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(fresh, (pool[p] for p in control))]
    print(f"control: cosine(cached, re-embedded) min {min(cos):.5f} mean {np.mean(cos):.5f}", flush=True)
    if min(cos) < 0.999:
        raise SystemExit("cached embeddings do not match this run's images")
    if need:
        for p, e in zip(need, encoder.encode_images([str(image_dir / p) for p in need], cfg.batch_size)):
            pool[p] = e
    print(f"embedded {len(need)} images no cache covered", flush=True)

    lf = Path("results") / f"lfcbm_seed{args.seed}"
    shared = lf / "lfcbm_shared_img_cache"
    shared.mkdir(parents=True, exist_ok=True)
    for split, files in splits.items():
        arr = np.stack([pool[p] for p in files]).astype(np.float32)
        np.save(shared / path_hash_name([str(image_dir / p) for p in files]), arr)
        if split != "test":
            for regime in REGIMES:
                d = lf / f"lfcbm_{regime}_cache"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / ("clip_img_train.npy" if split == "train" else "clip_img_valid.npy"), arr)
    grid_lf = args.grid_run[0] / "results" / f"lfcbm_seed{re.search(r'run_s(\d+)', str(args.grid_run[0])).group(1)}"
    for regime in REGIMES:
        shutil.copy2(grid_lf / f"lfcbm_{regime}_cache" / "clip_txt_concepts.npy", lf / f"lfcbm_{regime}_cache" / "clip_txt_concepts.npy")
    print("caches written:", sorted(str(p.relative_to(lf)) for p in lf.rglob("*.npy")), flush=True)


if __name__ == "__main__":
    main()
