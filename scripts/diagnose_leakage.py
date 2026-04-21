"""Diagnose concept leakage in CBM, CEM, and ProbCBM.

Implements three leakage measures from the literature:
1. CTL (Concept-Task Leakage) — Parisini et al., 2025
2. ICL (Interconcept Leakage) — Parisini et al., 2025
3. Conditional MI — Schoen et al., 2025

Usage::
    PYTHONPATH=. python scripts/diagnose_leakage.py
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mutual_info_score

from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load


def entropy_binary(y: np.ndarray) -> float:
    """Entropy of a discrete label array."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + 1e-12))


def mi_discrete(a: np.ndarray, b: np.ndarray) -> float:
    """MI between two discrete arrays."""
    return mutual_info_score(a.astype(int), b.astype(int))


def mi_continuous_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """MI between continuous x and discrete y via KSG estimator."""
    result = mutual_info_classif(
        x.reshape(-1, 1), y.astype(int), discrete_features=False, n_neighbors=5
    )
    return float(result[0])


def compute_ctl(
    c_hat: np.ndarray, c_gt: np.ndarray, y: np.ndarray, concepts: list[str]
) -> dict[str, float]:
    """Compute per-concept CTL scores."""
    H_y = entropy_binary(y)
    ctl = {}
    for i, name in enumerate(concepts):
        if i >= c_hat.shape[1]:
            continue
        mi_gt = mi_discrete(c_gt[:, i], y) / H_y
        mi_pred = mi_continuous_discrete(c_hat[:, i], y) / H_y
        ctl[name] = max(0.0, mi_pred - mi_gt)
    return ctl


def compute_icl(
    c_hat: np.ndarray, c_gt: np.ndarray, concepts: list[str]
) -> float:
    """Compute mean ICL score across concept pairs."""
    k = min(c_hat.shape[1], c_gt.shape[1])
    if k < 2:
        return 0.0

    icl_sum = 0.0
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            # GT NMI (discrete)
            mi_gt = mi_discrete(c_gt[:, i], c_gt[:, j])
            h_i = entropy_binary(c_gt[:, i])
            h_j = entropy_binary(c_gt[:, j])
            nmi_gt = mi_gt / (np.sqrt(h_i * h_j) + 1e-12)

            # Predicted NMI (continuous — discretize at 0.5 for tractability)
            c_hat_bin_i = (c_hat[:, i] >= 0.5).astype(int)
            c_hat_bin_j = (c_hat[:, j] >= 0.5).astype(int)
            mi_pred = mi_discrete(c_hat_bin_i, c_hat_bin_j)
            h_pi = entropy_binary(c_hat_bin_i)
            h_pj = entropy_binary(c_hat_bin_j)
            nmi_pred = mi_pred / (np.sqrt(h_pi * h_pj) + 1e-12)

            icl_sum += max(0.0, nmi_pred - nmi_gt)
            n_pairs += 1

    return icl_sum / max(n_pairs, 1)


def compute_conditional_mi(
    c_hat: np.ndarray, c_gt: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Compute conditional MI leakage: H(y|c) - H(y|c, c_hat)."""
    k = min(c_hat.shape[1], c_gt.shape[1])
    c_gt_k = c_gt[:, :k].astype(float)
    c_hat_k = c_hat[:, :k].astype(float)

    # H(y|c) via cross-entropy of probe on GT concepts
    lr_gt = LogisticRegression(max_iter=2000, C=1.0)
    lr_gt.fit(c_gt_k, y)
    proba_gt = lr_gt.predict_proba(c_gt_k)
    h_y_given_c = log_loss(y, proba_gt)

    # H(y|c, c_hat) via cross-entropy of probe on GT + predicted
    X_combined = np.concatenate([c_gt_k, c_hat_k], axis=1)
    lr_both = LogisticRegression(max_iter=2000, C=1.0)
    lr_both.fit(X_combined, y)
    proba_both = lr_both.predict_proba(X_combined)
    h_y_given_c_chat = log_loss(y, proba_both)

    leakage = max(0.0, h_y_given_c - h_y_given_c_chat)

    return {
        "H(y|c)": h_y_given_c,
        "H(y|c,c_hat)": h_y_given_c_chat,
        "leakage": leakage,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept-preset",
        default=None,
        choices=["foot_subtypes"],
        help="Use subconcept (human_concepts) dataset",
    )
    args = parser.parse_args()

    if args.concept_preset == "foot_subtypes":
        config = RobotBenchmarkConfig.default_subconcept()
    else:
        config = RobotBenchmarkConfig(seed=1014)
    config.seed = 1014

    data = load(config.get_dataset_path())
    test = data.test
    y = test.y.astype(int)
    concepts = list(test.concepts)
    c_gt = test.C[:, : len(concepts)].astype(int)
    print(f"Dataset: {len(concepts)} concepts, {len(y)} test samples")

    # Load models
    models = {}
    for family in ["cbm", "cem", "probcbm", "ecbm"]:
        try:
            cfg_family = RobotBenchmarkConfig.default_subconcept() if args.concept_preset else RobotBenchmarkConfig(seed=1014)
            cfg_family.seed = 1014
            cfg_family.cbm_family = family
            model_path = cfg_family.get_model_path(family)
            if not model_path.exists():
                print(f"Skipping {family} — model not found at {model_path}")
                continue
            models[family.upper()] = load(model_path)
        except Exception as e:
            print(f"Skipping {family} — {e}")

    # Extract concept predictions
    concept_preds = {}
    for name, model in models.items():
        concept_preds[name] = model.concept_detector.predict_proba(test)

    # === CTL ===
    print("=== CTL (Concept-Task Leakage) — per concept ===")
    print(f"{'Concept':>15s}", end="")
    for mname in models:
        print(f"  {mname:>8s}", end="")
    print()

    ctl_scores = {}
    for mname in models:
        ctl_scores[mname] = compute_ctl(concept_preds[mname], c_gt, y, concepts)

    for cname in concepts:
        print(f"{cname:>15s}", end="")
        for mname in models:
            val = ctl_scores[mname].get(cname, 0)
            print(f"  {val:>8.4f}", end="")
        print()

    print(f"{'MEAN':>15s}", end="")
    for mname in models:
        vals = list(ctl_scores[mname].values())
        print(f"  {np.mean(vals):>8.4f}", end="")
    print("\n")

    # === ICL ===
    print("=== ICL (Interconcept Leakage) ===")
    for mname in models:
        icl = compute_icl(concept_preds[mname], c_gt, concepts)
        print(f"  {mname:>8s}: {icl:.4f}")
    print()

    # === Conditional MI ===
    print("=== Conditional MI Leakage ===")
    print(f"{'Model':>8s}  {'H(y|c)':>8s}  {'H(y|c,c_hat)':>14s}  {'Leakage':>8s}")
    for mname in models:
        cmi = compute_conditional_mi(concept_preds[mname], c_gt, y)
        print(
            f"{mname:>8s}  {cmi['H(y|c)']:>8.4f}  {cmi['H(y|c,c_hat)']:>14.4f}  {cmi['leakage']:>8.4f}"
        )
    print()

    # === Summary ===
    print("=== Per-concept detection accuracy (for context) ===")
    print(f"{'Concept':>15s}", end="")
    for mname in models:
        print(f"  {mname:>8s}", end="")
    print()
    for i, cname in enumerate(concepts):
        print(f"{cname:>15s}", end="")
        for mname in models:
            cp = concept_preds[mname]
            if i < cp.shape[1]:
                acc = ((cp[:, i] >= 0.5).astype(int) == c_gt[:, i]).mean()
                print(f"  {acc:>7.1%}", end="")
            else:
                print(f"  {'—':>8s}", end="")
        print()

    # === LaTeX table row output ===
    print("\n=== LaTeX table rows (for Appendix Table::Leakage) ===")
    icl_scores = {}
    cmi_scores = {}
    ctl_means = {}
    for mname in models:
        icl_scores[mname] = compute_icl(concept_preds[mname], c_gt, concepts)
        cmi_scores[mname] = compute_conditional_mi(concept_preds[mname], c_gt, y)
        ctl_means[mname] = np.mean(list(ctl_scores[mname].values()))

    for mname in models:
        ctl_val = ctl_means[mname]
        icl_val = icl_scores[mname]
        cmi_val = cmi_scores[mname]["leakage"]
        print(f"\\{mname.replace('PROBCBM','ProbCBM').replace('ECBM','ECBM')}{{}}"
              f" & {ctl_val:.4f} & {icl_val:.4f} & {cmi_val:.4f} \\\\")


if __name__ == "__main__":
    main()
