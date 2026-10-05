"""Run the robot pipeline with ProbCBM's concept prototypes held fixed and opposite (causal test).

In the trained grid models, the learned "present" and "absent" prototypes of MouthType and HasKnees end up nearly
identical, and runs whose label head misses one of them ignore interventions (scripts/paper/measure_probcbm_heads.py).
With PROBCBM_FIXED_PROTOTYPES=1, every concept's absent prototype is set to minus its present prototype at
initialization and both are frozen; everything else is the official ProbCBM. With PROBCBM_VIB_BETA=0 the
penalty pulling concept embeddings toward the prior (vib_beta, default 5e-5) is switched off. Without either
variable this is the unmodified pipeline (control). Arguments are passed to scripts/robot_pipeline.py.

    PROBCBM_FIXED_PROTOTYPES=1 PYTHONPATH=. python scripts/paper/train_probcbm_fixed_prototypes.py --seed 1014 ...
"""

import importlib
import os
import runpy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.baselines._common import require_cem_dependencies

PIPELINE = Path(__file__).resolve().parents[1] / "robot_pipeline.py"
FIXED_PROTOTYPES = os.environ.get("PROBCBM_FIXED_PROTOTYPES") == "1"
VIB_BETA = os.environ.get("PROBCBM_VIB_BETA")


def patch_probcbm() -> None:
    """Fix the concept prototypes and/or set vib_beta on the cem package's ProbCBM, as the environment asks."""
    if not FIXED_PROTOTYPES and VIB_BETA is None:
        return
    require_cem_dependencies()  # puts the cem checkout on the import path
    module = importlib.import_module("cem.models.probcbm")
    assert require_cem_dependencies().ProbCBM is module.ProbCBM
    original_init = module.ProbCBM.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if FIXED_PROTOTYPES:
            with torch.no_grad():
                present = F.normalize(self.concept_vectors[1], p=2, dim=-1)
                self.concept_vectors[1].copy_(present)
                self.concept_vectors[0].copy_(-present)
            self.concept_vectors.requires_grad_(False)
        if VIB_BETA is not None:
            self.loss_concept.vib_beta = float(VIB_BETA)

    module.ProbCBM.__init__ = patched_init
    print(
        f"ProbCBM patched: fixed prototypes={FIXED_PROTOTYPES}, vib_beta={VIB_BETA}",
        flush=True,
    )


if __name__ == "__main__":
    patch_probcbm()
    sys.argv = [str(PIPELINE), *sys.argv[1:]]
    runpy.run_path(str(PIPELINE), run_name="__main__")
