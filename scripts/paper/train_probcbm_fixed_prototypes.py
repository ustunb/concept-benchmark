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

import torch
import torch.nn.functional as F

FIXED_PROTOTYPES = os.environ.get("PROBCBM_FIXED_PROTOTYPES") == "1"
VIB_BETA = os.environ.get("PROBCBM_VIB_BETA")
if FIXED_PROTOTYPES or VIB_BETA is not None:
    try:  # the pipeline puts its cem checkout on the import path when loading the dependencies
        from experiments.baselines._common import require_cem_dependencies
    except ImportError:
        from experiments.cem_integration import require_cem_dependencies
    require_cem_dependencies()
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

sys.argv = ["scripts/robot_pipeline.py", *sys.argv[1:]]
runpy.run_path("scripts/robot_pipeline.py", run_name="__main__")
