"""Make LFCBM models trained on DSMLP usable in another run directory without recomputing CLIP embeddings.

The LFCBM models store their cache folder as a DSMLP path, and the shared CLIP image cache is keyed by a hash of
the image path strings (experiments/lfcbm.py, `transform`). For one seed's run directory this
  * points every LFCBM model's `cfg.cache_dir` at the run's own lfcbm_seed<seed> folder, and
  * copies each cached image embedding under the hash of the paths this run will pass
    (`data_dir / "robot_images" / file`, scripts/robot_pipeline.py `_prepare_lfcbm_labeled_data`),
so the LFCBM concepts are exactly the ones computed on DSMLP. Fails if an expected cache file is missing.

    cd <run dir> && PYTHONPATH=. python repoint_lfcbm_cache.py --seed 1015 --old-root /home/jskirzynski/cb-run-s1015
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.paths import data_dir


def cache_name(paths: list[str]) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(str(p).encode("utf-8"))
    h.update(str(len(paths)).encode("utf-8"))
    return f"clip_img_{h.hexdigest()[:16]}.npy"


def repoint(obj, cache_root: Path, seen=None) -> int:
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, dict):
        return sum(repoint(v, cache_root, seen) for v in obj.values())
    if not hasattr(obj, "__dict__"):
        return 0
    n = 0
    cfg = getattr(obj, "cfg", None)
    if cfg is not None and getattr(cfg, "cache_dir", None) is not None:
        cfg.cache_dir = cache_root / Path(cfg.cache_dir).name
        n += 1
    for v in vars(obj).values():
        n += repoint(v, cache_root, seen)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--old-root", required=True, help="Repository folder the LFCBMs were trained in (DSMLP).")
    args = ap.parse_args()
    s = args.seed
    results = Path("results")
    cache_root = (results / f"lfcbm_seed{s}").resolve()
    shared = cache_root / "lfcbm_shared_img_cache"

    for f in sorted(results.glob(f"*lfcbm_*_seed{s}.model")):
        model = load(f)
        n = repoint(model, cache_root)
        save(model, f, overwrite=True)
        print(f"{f.name}: cache_dir repointed ({n})")

    data = load(results / f"robot_image_4_subconcept_seed{s}.data")
    old_dir = f"{args.old_root}/data/robot_images"
    new_dir = data_dir / "robot_images"
    for split in ("train", "validation", "test"):
        files = getattr(data, split).inputs
        old = shared / cache_name([f"{old_dir}/{p}" for p in files])
        new = shared / cache_name([str(new_dir / p) for p in files])
        if not old.exists():
            raise SystemExit(f"{split}: expected cached embeddings {old.name} not found")
        if not new.exists():
            shutil.copy2(old, new)
        print(f"{split}: {old.name} -> {new.name}")


if __name__ == "__main__":
    main()
