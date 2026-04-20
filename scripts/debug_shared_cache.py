"""Debug why shared CLIP cache isn't being hit."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, ".")
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load

config = RobotBenchmarkConfig.default_subconcept()
config.seed = 1014
data = load(config.get_dataset_path())
image_dir = Path("data/robot_images")

# Hash that transform() computes
train_paths = [str(image_dir / p) for p in data.train.X]
h = hashlib.sha1()
for p in train_paths:
    h.update(p.encode("utf-8"))
h.update(str(len(train_paths)).encode("utf-8"))
train_hash = h.hexdigest()[:16]
print(f"train hash from data: {train_hash}")
print(f"train paths[0]: {train_paths[0]}")
print(f"train count: {len(train_paths)}")

# Shared cache files
shared = Path("results/lfcbm_shared_img_cache")
print(f"\nShared cache files:")
for f in sorted(shared.iterdir()):
    print(f"  {f.name}")

# LFCBM cache_dir
lf = load("results/robot_image_stochastic_4_subconcept_lfcbm_machine.model")
print(f"\nLFCBM cfg.cache_dir: {lf.cfg.cache_dir}")
cache_dir_path = Path(str(lf.cfg.cache_dir))
shared_from_code = cache_dir_path.parent / "lfcbm_shared_img_cache"
print(f"shared_dir resolved: {shared_from_code}")
print(f"shared_dir exists: {shared_from_code.exists()}")

target = shared_from_code / f"clip_img_{train_hash}.npy"
print(f"target file: {target}")
print(f"target exists: {target.exists()}")

# Also check: does the absolute path matter?
print(f"\nAbsolute shared: {shared.resolve()}")
print(f"Absolute from code: {shared_from_code.resolve()}")
print(f"Match: {shared.resolve() == shared_from_code.resolve()}")
