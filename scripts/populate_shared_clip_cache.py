"""Copy existing CLIP image embeddings into the shared cache directory."""
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load

shared = Path("results/lfcbm_shared_img_cache")
shared.mkdir(parents=True, exist_ok=True)

CACHE_DIRS = ["lfcbm_cache", "lfcbm_machine_cache", "lfcbm_llm_cache", "lfcbm_clip_cache"]

for label, config_fn in [
    ("subconcept", RobotBenchmarkConfig.default_subconcept),
    ("ideal", lambda: RobotBenchmarkConfig(seed=1014)),
]:
    config = config_fn()
    config.seed = 1014
    data = load(config.get_dataset_path())
    from concept_benchmark.paths import data_dir
image_dir = data_dir / "robot_images"

    for split_name, file_name, sample in [
        ("train", "train", data.train),
        ("validation", "valid", data.validation),
        ("test", "test", data.test),
    ]:
        paths = [str(image_dir / p) for p in sample.X]
        h = hashlib.sha1()
        for p in paths:
            h.update(p.encode("utf-8"))
        h.update(str(len(paths)).encode("utf-8"))
        dest = shared / f"clip_img_{h.hexdigest()[:16]}.npy"

        if dest.exists():
            print(f"{label}/{split_name}: EXISTS {dest.name}")
            continue

        for rc in CACHE_DIRS:
            src = Path(f"results/{rc}/clip_img_{file_name}.npy")
            if src.exists():
                arr = np.load(src)
                if arr.shape[0] == len(paths):
                    shutil.copy2(src, dest)
                    print(f"{label}/{split_name}: COPIED {src} -> {dest.name} ({arr.shape})")
                    break
        else:
            print(f"{label}/{split_name}: NOT FOUND ({len(paths)} rows)")

print("\nShared cache:")
for f in sorted(shared.iterdir()):
    print(f"  {f.name} ({f.stat().st_size // 1024} KB)")
