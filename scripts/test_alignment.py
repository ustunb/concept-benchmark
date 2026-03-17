#!/usr/bin/env python
"""Quick alignment test — subconcept only, k=0 and k=3."""
import copy, sys
from concept_benchmark.utils import patch_macos_dataloader, compute_accuracy, determine_device, get_loader_config
import robot_pipeline as robot
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.alignment import align_frontend_weights

patch_macos_dataloader()

cfg = RobotBenchmarkConfig(
    seed=1014, subconcept=True,
    intervention_budgets=[3], intervention_thresholds=[0.2],
    intervention_regimes=["baseline"], intervention_strategy="kflip",
    alignment_constraints={"has_knees": 1},
)
data = robot.setup_dataset(cfg)
cbm = robot.train_cbm(cfg, data)
cbm_k0 = float((cbm.predict(data.test) == data.test.y).mean())
print(f"CBM k=0: {cbm_k0:.4f}  (paper: 0.7812)")

align_data = robot.align(cfg, cbm, data)

aligned_k0 = float(align_data["aligned_accuracy"])
print(f"Aligned k=0: {aligned_k0:.4f}  (paper: 0.7656)")

# Aligned k=3
aligned_fe = copy.deepcopy(cbm.front_end_model)
aligned_fe = align_frontend_weights(aligned_fe, list(data.test.concepts), align_data["aligned_weights"])
c_preds = cbm.concept_detector.predict(data.test)
isettings = robot.InterventionSettings(seed=cfg.seed, budgets=[3], intervention_accuracy=cfg.intervention_accuracy, intervention_threshold=0.2)
_, _, int_results = robot._test_interventions(prob_test=c_preds, settings=isettings, acc_det=aligned_k0, fe=aligned_fe, test=data.test)
aligned_k3 = float(next(iter(int_results.values()))["accuracy"])
dnn_gain = aligned_k3 - 0.8746
print(f"Aligned k=3: {aligned_k3:.4f}  gain vs DNN: {dnn_gain:+.1%}  (paper: -8.0%)")

if abs(aligned_k0 - 0.7656) < 0.001 and abs(dnn_gain - (-0.080)) < 0.01:
    print("PASS")
else:
    print("FAIL")
    sys.exit(1)
