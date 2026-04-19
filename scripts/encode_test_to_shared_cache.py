"""Encode test images and save to shared CLIP cache."""
import hashlib
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, ".")
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load

config = RobotBenchmarkConfig.default_subconcept()
config.seed = 1014
data = load(config.get_dataset_path())
image_dir = Path("data/robot_images")

shared = Path("results/lfcbm_shared_img_cache")
shared.mkdir(parents=True, exist_ok=True)

test_paths = [str(image_dir / p) for p in data.test.X]
h = hashlib.sha1()
for p in test_paths:
    h.update(p.encode("utf-8"))
h.update(str(len(test_paths)).encode("utf-8"))
dest = shared / f"clip_img_{h.hexdigest()[:16]}.npy"

if dest.exists():
    print(f"Already exists: {dest}")
else:
    lf = load("results/robot_image_stochastic_4_subconcept_lfcbm.model")
    print(f"Encoding {len(test_paths)} test images...", flush=True)
    img = lf._get_encoder().encode_images(test_paths, lf.cfg.batch_size)
    np.save(dest, img)
    print(f"Saved: {dest} {img.shape}", flush=True)

print("\nShared cache:")
for f in sorted(shared.iterdir()):
    a = np.load(f)
    print(f"  {f.name}: {a.shape}")
