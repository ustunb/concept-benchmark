"""Robot classification benchmark pipeline.

Provides functions to run each stage of the robot benchmark programmatically.

Usage:
    python scripts/robot_pipeline.py --seed 1014
    python scripts/robot_pipeline.py --seed 1014 --subconcept
    python scripts/robot_pipeline.py --seed 1014 --subconcept --regimes baseline expert
    python scripts/robot_pipeline.py --config my_config.yaml
"""
from __future__ import annotations

import copy
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from concept_benchmark.utils import (
    compute_accuracy,
    create_skewed_splits_full,
    determine_device,
    get_loader_config,
    patch_macos_dataloader,
    run_alignment,
    set_deterministic_seed,
)
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.models import (
    ConceptBasedModel,
    ConceptDetector,
    FrontEndModel,
    RobotClassifierCNN,
    RobotConceptClassifier,
)
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot import create_synthetic_dataset

@dataclass
class InterventionSettings:
    """Typed config for _test_interventions, replacing the old ``sttngs`` dict."""

    seed: int
    budgets: List[int]
    intervention_accuracy: float = 0.9
    intervention_threshold: float = 1.0
    intervention_strategy: str = "kflip"
    intervention_expert: str = ""  # "" for standard path, "llm" for inline LLM
    intervention_llm: Optional[Dict[str, Any]] = None
    run_dir: str = "."


class FEOnProbs(FrontEndModel):
    """Wrap an LFCBM sklearn classifier so it works as a FrontEndModel.

    Applies logit transform before predict_proba, matching how the
    LFCBM classifier was trained (on z-scored, then logit-transformed features).
    """
    _kflip_fast_path = False  # class-level: fast path skips our logit transform

    def __init__(self, clf):
        super().__init__()
        self.model = clf

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        P = np.clip(P, 1e-6, 1 - 1e-6)
        Z = np.log(P / (1.0 - P))
        return self.model.predict_proba(Z)


_set_deterministic_seed = set_deterministic_seed  # backward compat alias


# Lazy import to avoid circular deps — intervention modules
_intervention_imported = False


def _ensure_intervention_imports():
    global _intervention_imported
    if not _intervention_imported:
        global ConceptInterventionRunner, InterventionConfig, KFlipInterventionStrategy
        from concept_benchmark.intervention import (
            ConceptInterventionRunner,
            InterventionConfig,
        )
        from concept_benchmark.kflip import KFlipInterventionStrategy

        _intervention_imported = True


# ── Stage: setup_dataset ──────────────────────────────────────────────

def setup_dataset(config: RobotBenchmarkConfig):
    """Generate robot dataset, apply skewed splits, and save.

    Returns the saved ConceptDataset.
    """
    settings = config.to_dict()
    logger.info("Generating robot dataset...")
    data = create_synthetic_dataset(**settings)
    data.generate_cvindices(seed=config.seed)

    rng = np.random.default_rng(config.seed)
    sk_data = create_skewed_splits_full(dataset=data, rng=rng, **settings)
    save(sk_data, config.get_dataset_path(), overwrite=True)
    return sk_data


# ── Stage: train_cbm ──────────────────────────────────────────────────

def train_cbm(
    config: RobotBenchmarkConfig,
    data=None,
    save_key: Optional[str] = "cbm",
) -> ConceptBasedModel:
    """Train a ConceptBasedModel (concept detector + frontend).

    Args:
        save_key: Model save key. Use None to skip saving.

    Returns the trained CBM.
    """
    _set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    # Apply concept missingness if configured
    if config.concept_missing_mech != "none":
        if config.concept_missing <= 0.0:
            raise ValueError(
                "concept_missing must be > 0 when concept_missing_mech is not 'none'"
            )
        data.sample_concept_missingness(
            p=config.concept_missing,
            mechanism=config.concept_missing_mech,
            rng=np.random.default_rng(config.seed),
        )
        data.training.concept_missing = True

    _macos = platform.system() == "Darwin"
    loader_config = {
        "device": device,
        "batch_size": config.batch_size,
        "num_workers": 0 if _macos else 12,
        "pin_memory": not _macos,
    }
    torch.manual_seed(config.seed)

    cd = ConceptDetector(
        model=RobotConceptClassifier(
            num_concepts=data.training.n_concepts,
            input_size=config.input_size,
        )
    )
    cbm = ConceptBasedModel(concept_detector=cd)
    cbm.fit(
        train_dataset=data.training,
        valid_dataset=data.validation,
        freeze=False,
        concept_embed_params={"shuffle": False, **loader_config},
        fit_params={
            "epochs": config.epochs,
            "lr": config.lr,
            "patience": config.patience,
            **loader_config,
        },
    )

    test_pred = cbm.predict(data.test)
    logger.info("Test Accuracy: %s", np.mean(test_pred == data.test.y))

    if save_key is not None:
        save(cbm, config.get_model_path(save_key), overwrite=True)
    return cbm


def _build_concept_groups(concept_names, concept_spec):
    """Build groups of concept column indices from the concept spec.

    For each key in ``concept_spec``, find all columns in ``concept_names``
    whose name equals or starts with ``key_``.  Any ungrouped columns get
    their own singleton group.

    Returns dict mapping group name → list of column indices.
    """
    groups = {}
    used = set()
    for base in list(concept_spec.keys()):
        idxs = [i for i, c in enumerate(concept_names)
                if c == base or c.startswith(f"{base}_")]
        if idxs:
            groups[base] = idxs
            used.update(idxs)
    for i, c in enumerate(concept_names):
        if i not in used:
            groups[c] = [i]
    return groups


def _flip_onehot_row(row, idxs, rng):
    """Flip one active bit in a one-hot group to a random other option."""
    vals = row[idxs]
    if len(idxs) == 1:
        row[idxs[0]] = 1.0 - row[idxs[0]]
        return
    s = int(vals.sum())
    if s != 1:
        return
    active = int(np.argmax(vals))
    choices = [j for j in range(len(idxs)) if j != active]
    if not choices:
        return
    new_j = int(rng.choice(choices))
    row[idxs] = 0.0
    row[idxs[new_j]] = 1.0


def _apply_concept_noise_grouped(C, concept_names, concept_spec, rate, rng):
    """Apply group-level one-hot flip noise (original paper method).

    For each sample and each concept group, with probability ``rate``,
    randomly switch the active category to a different one within the group.
    """
    C_out = C.astype(np.float32).copy()
    groups = _build_concept_groups(concept_names, concept_spec)
    for r in range(C_out.shape[0]):
        for _, idxs in groups.items():
            if rng.random() < float(rate):
                _flip_onehot_row(C_out[r], idxs, rng)
    return C_out


def _clone_sample_with_C(sample, C_new):
    """Clone a dataset sample with replaced concept matrix."""
    return sample.__class__(
        parent=sample.parent,
        X=sample.X,
        C=C_new.astype(np.float32),
        y=sample.y,
        meta=sample.meta,
        transform=sample.transform,
        concept_transform=sample.concept_transform,
        target_transform=sample.target_transform,
        base_dir=getattr(sample, "base_dir", None),
    )


def train_cbm_subjective(
    config: RobotBenchmarkConfig,
    data=None,
) -> ConceptBasedModel:
    """Train a CBM on noisy (subjective) concept labels.

    Uses group-level one-hot flip noise (matching the original paper):
    for each sample and each concept group, with probability
    ``config.subjective_noise_rate``, randomly switch the active
    category to a different one within the group.

    The noisy CBM is saved to ``config.get_model_path("cbm_subjective")``.
    """
    if data is None:
        data = load(config.get_dataset_path())
    noisy_data = copy.deepcopy(data)

    # Offset seed so noise RNG is independent of data-generation RNG.
    rng = np.random.default_rng(config.seed + 555)
    concept_names = list(noisy_data.training.concepts)
    concept_spec = config.concepts

    # Apply group-level noise to training and validation splits
    noisy_data.training = _clone_sample_with_C(
        noisy_data.training,
        _apply_concept_noise_grouped(
            noisy_data.training.C, concept_names, concept_spec,
            config.subjective_noise_rate, rng,
        ),
    )
    if hasattr(noisy_data, "validation") and noisy_data.validation is not None:
        noisy_data.validation = _clone_sample_with_C(
            noisy_data.validation,
            _apply_concept_noise_grouped(
                noisy_data.validation.C, concept_names, concept_spec,
                config.subjective_noise_rate, rng,
            ),
        )

    cbm = train_cbm(config, data=noisy_data, save_key=None)
    save(cbm, config.get_model_path("cbm_subjective"), overwrite=True)
    return cbm


def train_lfcbm(
    config: RobotBenchmarkConfig,
    data=None,
) -> dict:
    """Train a Label-Free CBM using CLIP embeddings.

    Requires ``open-clip-torch`` and a concepts JSONL file.
    Saves to ``config.get_model_path("lfcbm")``.
    """
    from concept_benchmark.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig

    _set_deterministic_seed(config.seed)

    if data is None:
        data = load(config.get_dataset_path())

    concepts_file = config.lfcbm_concepts_file
    if not concepts_file:
        from concept_benchmark.paths import pkg_dir
        suffix = "_subconcept" if config.subconcept else ""
        default = pkg_dir / "concept_descriptions" / f"gt_concepts{suffix}.jsonl"
        if default.exists():
            concepts_file = str(default)
        else:
            raise FileNotFoundError(
                f"No LFCBM concepts file: set config.lfcbm_concepts_file or create {default}"
            )

    concept_set = LFConceptSet.from_file(concepts_file)
    device_str = str(determine_device())

    cfg = LFTrainingConfig(
        device=device_str,
        seed=config.seed,
        cache_dir=config.get_model_path("lfcbm").parent / "lfcbm_cache",
    )
    lfcbm = LabelFreeCBM(cfg)

    # Extract image paths from dataset
    train_paths = [str(p) for p in data.training.X]
    valid_paths = [str(p) for p in data.validation.X]

    stats = lfcbm.fit(
        train_X=train_paths,
        train_y=data.training.y.astype(int),
        valid_X=valid_paths,
        valid_y=data.validation.y.astype(int),
        concept_set=concept_set,
    )
    logger.info("LFCBM stats: %s/%s concepts kept", stats.get("kept_concepts"), stats.get("total_concepts"))

    out_dir = str(config.get_model_path("lfcbm")) + "_bundle"
    lfcbm.save(out_dir)
    bundle = {"lfcbm": lfcbm, "frontend": FEOnProbs(lfcbm.classifier)}
    save(bundle, config.get_model_path("lfcbm"), overwrite=True)
    return stats


# ── Stage: train_dnn ──────────────────────────────────────────────────

def train_dnn(
    config: RobotBenchmarkConfig,
    data=None,
) -> dict:
    """Train an end-to-end DNN baseline.

    Returns the best state_dict.
    """
    _set_deterministic_seed(config.seed)
    patch_macos_dataloader()
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    torch.manual_seed(config.seed)
    model = RobotClassifierCNN(input_size=config.input_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loader_config = get_loader_config(device)
    train_loader = data.training.loader(shuffle=True, **loader_config)
    valid_loader = data.validation.loader(shuffle=False, **loader_config)
    test_loader = data.test.loader(shuffle=False, **loader_config)

    model.to(device)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        model.train()
        for X, _, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for X, _, y in valid_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                batch_loss = criterion(outputs.squeeze(), y.float())
                val_loss_sum += batch_loss.item()
                val_batches += 1
        avg_val_loss = val_loss_sum / max(val_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if config.patience > 0 and epochs_no_improve >= config.patience:
                logger.info(
                    "Early stopping at epoch %d with best val loss %.6f",
                    epoch + 1, best_val_loss,
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_acc = compute_accuracy(model, train_loader, device=device)
    valid_acc = compute_accuracy(model, valid_loader, device=device)
    test_acc = compute_accuracy(model, test_loader, device=device)
    logger.info("Training Accuracy: %.2f%%", train_acc * 100)
    logger.info("Validation Accuracy: %.2f%%", valid_acc * 100)
    logger.info("Test Accuracy: %.2f%%", test_acc * 100)

    weights = best_state_dict if best_state_dict is not None else model.state_dict()
    # Move tensors to CPU before saving for cross-device portability
    weights = {k: v.cpu() for k, v in weights.items()}
    save(weights, config.get_model_path("dnn"), overwrite=True)
    return weights


# ── Intervention helper ───────────────────────────────────────────────

def _test_interventions(prob_test, settings: InterventionSettings, acc_det, fe, test, concept_names=None):
    """Run interventions for each budget and return results dict.

    Matches the original ``test_interventions`` from ``robot_concept_regimes.py``,
    including the inline LLM path (compute-once at K=max, batched multi-image calls,
    JSONL cache, flip-effect ranking, per-budget mask derivation).
    """
    import hashlib
    import json
    import time
    from pathlib import Path
    from types import SimpleNamespace

    _ensure_intervention_imports()

    intervention_results = {}
    rng = np.random.default_rng(settings.seed)
    budgets = list(settings.budgets)
    human_acc = settings.intervention_accuracy
    err_prob = 1.0 - human_acc

    _coerce_to_gt = concept_names is None
    if concept_names is None:
        concept_names = list(getattr(test, "concepts", []))
    else:
        concept_names = list(concept_names)

    # Coerce concept_proba to match dataset concept ground truth shape,
    # but only when the caller didn't supply its own concept space
    # (LLM/CLIP regimes operate in the LFCBM concept space which may
    # differ from GT dimensions).
    if _coerce_to_gt and hasattr(test, "C"):
        n_gt = int(test.C.shape[1])
        n_pred = int(prob_test.shape[1])
        if n_pred != n_gt:
            if n_pred > n_gt:
                prob_test = prob_test[:, :n_gt]
            else:
                pad = np.zeros((prob_test.shape[0], n_gt - n_pred), dtype=prob_test.dtype)
                prob_test = np.concatenate([prob_test, pad], axis=1)
        if len(concept_names) != prob_test.shape[1]:
            concept_names = concept_names[: prob_test.shape[1]]

    # Create a CBM wrapper for the intervention framework
    cbm = ConceptBasedModel(concept_detector=None, front_end_model=fe)
    runner = ConceptInterventionRunner(cbm)
    llm_cache = None
    _intervention_cache = None

    for budget in budgets:
        if int(budget) <= 0:
            key = f"top_{budget}_human_acc_{int(human_acc * 100)}"
            n_samples = prob_test.shape[0]
            intervention_results[key] = {
                "accuracy": float(acc_det),
                "accuracy_gain": 0.0,
                "predictions_intervened_on": 0,
                "predictions_changed": 0,
                "interventions_rate": 0.0,
                "intervention_rate": 0.0,
                "avg_edits_per_intervention": 0.0,
                "total_concept_checks": 0,
                "total_concept_confirmations": 0,
                "total_concept_edits_made": 0,
                "concepts_intervened": {},
                "concepts_edits": {},
            }
            continue

        config = InterventionConfig(
            max_concepts_per_instance=budget,
            random_state=settings.seed,
            score_threshold=settings.intervention_threshold,
            noise=1.0 - human_acc,
        )

        strategy = KFlipInterventionStrategy(
            exact_k=(settings.intervention_strategy == "exact_k"),
        )

        if settings.intervention_expert.lower() == "llm":
            # ── Inline LLM path (matching original robot_concept_regimes.py) ──
            try:
                from google.api_core.exceptions import ResourceExhausted
            except ImportError:
                ResourceExhausted = None

            from concept_benchmark.llm_client import make_llm_client

            llm_cfg = settings.intervention_llm or {}
            provider = str(llm_cfg.get("provider", "gemini"))
            model_name = str(llm_cfg.get("model", "gemini-2.5-flash-lite"))
            api_key_env = str(llm_cfg.get("api_key_env", "GEMINI_API_KEY"))

            import os
            api_key = str(llm_cfg.get("api_key", "")) or os.environ.get(api_key_env, "")
            if not api_key:
                raise SystemExit(
                    f"missing API key: set llm_api_key in config or {api_key_env} in env"
                )

            client = make_llm_client(provider, model_name, api_key)

            def _llm_call_with_retry(fn, *, max_retries=5, backoff=30.0, label="LLM"):
                """Call *fn* and retry on ResourceExhausted with exponential back-off."""
                for attempt in range(1, max_retries + 1):
                    try:
                        return fn()
                    except Exception as e:
                        if ResourceExhausted is not None and isinstance(e, ResourceExhausted):
                            logger.warning(
                                "%s ResourceExhausted attempt %d/%d: %s",
                                label, attempt, max_retries, e,
                            )
                            if attempt >= max_retries:
                                logger.error("%s giving up after %d attempts.", label, max_retries)
                                raise
                            time.sleep(backoff)
                        else:
                            raise

            def _resolve_img_path(i: int) -> str:
                p = Path(str(test.X[i]))
                base = getattr(test, "base_dir", Path("."))
                q = p if p.is_absolute() else (base / p)
                return str(q.resolve())

            def _llm_judge(image_path: str, names: list) -> dict:
                prompt = (
                    "You will be shown one robot image. "
                    "For each concept below, output 0 or 1 indicating ABSENT(0) or PRESENT(1). "
                    "Return ONLY one JSON object with string keys and 0/1 integer values.\n\n"
                    "concepts:\n- " + "\n- ".join(names) + "\n\n"
                    "Respond like: {\"conceptA\":1,\"conceptB\":0}"
                )
                logger.debug("LLM fallback judge start image=%s, concepts=%d", image_path, len(names))
                raw = _llm_call_with_retry(
                    lambda: (client.generate(prompt, [image_path]) or "").strip(),
                    label="LLM single-image",
                )
                parsed: dict = {}
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(v, bool):
                                parsed[str(k)] = 1 if v else 0
                            elif isinstance(v, (int, float, str)):
                                s = str(v).strip().lower()
                                parsed[str(k)] = 1 if s in {"1", "true", "yes", "present"} else 0
                except Exception:
                    pass
                return parsed

            def _llm_judge_batch(image_paths: list, per_image_names: list) -> list:
                N = len(image_paths)
                lines = []
                for i, names in enumerate(per_image_names):
                    lines.append(f"Image {i}: " + ", ".join(names))

                prompt = (
                    f"You will be shown {N} robot image(s) in order. "
                    f"For each image i (0..{N - 1}), output ONLY a JSON array of length {N} "
                    "where array[i] is a JSON object mapping the listed concepts to 0/1 integers "
                    "(ABSENT=0, PRESENT=1). No Markdown code fences, no extra keys, "
                    "and no text outside the JSON.\n\n"
                    "Per-image concepts:\n- " + "\n- ".join(lines)
                )

                raw = (client.generate(prompt, image_paths) or "").strip()

                # Strip Markdown ``` fences if the model ignores instructions
                if raw.startswith("```"):
                    fence_lines = raw.splitlines()
                    fence_lines = fence_lines[1:]
                    if fence_lines and fence_lines[-1].strip().startswith("```"):
                        fence_lines = fence_lines[:-1]
                    raw_clean = "\n".join(fence_lines).strip()
                else:
                    raw_clean = raw

                def _to01(v):
                    if isinstance(v, bool):
                        return 1 if v else 0
                    s = str(v).strip().lower()
                    return 1 if s in {"1", "true", "yes", "present"} else 0

                out: list = [dict() for _ in range(N)]

                try:
                    obj = json.loads(raw_clean)
                except Exception as e:
                    logger.debug("json.loads failed in _llm_judge_batch: %r", e)
                    logger.debug(
                        "raw_clean (first 400 chars): %s",
                        raw_clean[:400].replace("\n", "\\n"),
                    )
                    return out

                if isinstance(obj, list):
                    if len(obj) == 1 and isinstance(obj[0], list):
                        arr = obj[0]
                    else:
                        arr = obj

                    for i in range(min(N, len(arr))):
                        d = arr[i]
                        if not isinstance(d, dict):
                            continue
                        allow = set(per_image_names[i])
                        for k, v in d.items():
                            if k in allow:
                                out[i][str(k)] = _to01(v)

                elif isinstance(obj, dict):
                    for i_str, d in obj.items():
                        try:
                            i = int(i_str)
                        except Exception:
                            continue
                        if not (0 <= i < N and isinstance(d, dict)):
                            continue
                        allow = set(per_image_names[i])
                        for k, v in d.items():
                            if k in allow:
                                out[i][str(k)] = _to01(v)

                return out

            # compute-once at K=max(budgets), batch LLM once, reuse for smaller budgets
            if llm_cache is None:
                batch = runner._build_batch(
                    dataset=test,
                    concept_proba=prob_test,
                    concept_true=np.full_like(prob_test, np.nan, dtype=np.float32),
                    labels=test.y.astype(int),
                    instance_ids=None,
                )
                max_budget = max(int(b) for b in budgets)
                maxK = int(min(max_budget, batch.C_pred.shape[1]))
                config_max = InterventionConfig(
                    max_concepts_per_instance=maxK,
                    random_state=settings.seed,
                    score_threshold=settings.intervention_threshold,
                    noise=1.0 - human_acc,
                )
                proposal_max = strategy.propose(cbm, batch, config_max)
                mask_max = proposal_max.mask

                C_before = batch.C_pred
                y_prob_before = fe.predict_proba((C_before >= 0.5).astype(int))

                C_true_llm = np.full_like(C_before, np.nan, dtype=float)

                # JSONL on-disk cache
                run_root = Path(settings.run_dir)
                cache_dir = run_root / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)

                def _concepts_sig():
                    h = hashlib.sha1()
                    for name in map(str, concept_names):
                        h.update(name.encode("utf-8"))
                        h.update(b"\x00")
                    return h.hexdigest()

                def _dataset_sig():
                    h = hashlib.sha1()
                    for pth in map(str, test.X):
                        h.update(pth.encode("utf-8"))
                        h.update(b"\x00")
                    return h.hexdigest()

                cache_path = cache_dir / f"llm_interventions_{_concepts_sig()}_{_dataset_sig()}.jsonl"

                def _load_cache():
                    d = {}
                    if cache_path.exists():
                        with open(cache_path, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    rec = json.loads(line)
                                    i0 = int(rec["i"])
                                    votes_idx = {int(k): int(v) for k, v in rec.get("votes_idx", {}).items()}
                                    d[i0] = votes_idx
                                except Exception:
                                    continue
                    return d

                def _flush_cache(d):
                    with open(cache_path, "w", encoding="utf-8") as f:
                        for i0, votes_idx in d.items():
                            f.write(json.dumps({"i": int(i0), "votes_idx": {str(k): int(v) for k, v in votes_idx.items()}}) + "\n")

                if _intervention_cache is None:
                    _intervention_cache = _load_cache()

                tasks = []
                total_pairs = 0
                missing_pairs = 0
                for i in range(C_before.shape[0]):
                    idxs = np.where(mask_max[i])[0]
                    if idxs.size == 0:
                        continue
                    total_pairs += int(idxs.size)
                    known = _intervention_cache.get(i, {})
                    for j in idxs:
                        if j in known:
                            C_true_llm[i, j] = float(known[j])
                    missing = [j for j in idxs if j not in known]
                    missing_pairs += len(missing)
                    if not missing:
                        continue
                    image_path = _resolve_img_path(i)
                    names = [str(concept_names[j]) for j in missing]
                    tasks.append((i, image_path, names, missing))

                cached_pairs = total_pairs - missing_pairs
                logger.info(
                    "LLM intervention selection: total=%d, from_cache=%d, "
                    "to_query=%d, images_needing_llm=%d",
                    total_pairs, cached_pairs, missing_pairs, len(tasks),
                )

                bs = int(llm_cfg.get("batch_size") or 32)
                if bs < 1:
                    bs = 1
                n_batches = (len(tasks) + bs - 1) // bs
                if n_batches > 0:
                    logger.info(
                        "LLM starting batched calls: %d images, batch_size=%d, n_batches=%d",
                        len(tasks), bs, n_batches,
                    )

                retry_backoff = float(
                    llm_cfg.get("retry_backoff") or 30.0
                )
                max_retries = int(
                    llm_cfg.get("max_retries") or 5
                )
                sleep_time = float(
                    llm_cfg.get("batch_sleep") or 5.0
                )

                for batch_idx, s in enumerate(range(0, len(tasks), bs), start=1):
                    chunk = tasks[s:s + bs]
                    image_paths = [p for (_i, p, _n, _j) in chunk]
                    per_image_names = [names for (_i, _p, names, _idxs) in chunk]

                    votes_list = _llm_call_with_retry(
                        lambda: _llm_judge_batch(image_paths, per_image_names),
                        max_retries=max_retries,
                        backoff=retry_backoff,
                        label=f"LLM batch {batch_idx}/{n_batches}",
                    )
                    logger.debug(
                        "LLM batch %d/%d ok; sleeping %.1fs to respect rate limits",
                        batch_idx, n_batches, sleep_time,
                    )
                    time.sleep(sleep_time)

                    for (i_idx, _pth, names, idxs), votes in zip(chunk, votes_list):
                        if i_idx not in _intervention_cache:
                            _intervention_cache[i_idx] = {}

                        for j, name in zip(idxs, names):
                            if name in votes:
                                v = 1 if votes[name] else 0
                                C_true_llm[i_idx, j] = float(v)
                                _intervention_cache[i_idx][j] = v

                    _flush_cache(_intervention_cache)
                    logger.debug("LLM batch %d/%d complete; cache flushed.", batch_idx, n_batches)

                if n_batches > 0:
                    logger.info("LLM all %d batches complete.", n_batches)

                # rank concepts per instance by single-bit flip effect (reuse for budgets < K)
                order = [np.array([], dtype=int)] * C_before.shape[0]
                for i in range(C_before.shape[0]):
                    sel = np.where(mask_max[i])[0]
                    if sel.size == 0:
                        order[i] = np.array([], dtype=int)
                        continue
                    base_vec = (C_before[i] >= 0.5).astype(int)
                    base_prob = fe.predict_proba(base_vec[None, :])[0]
                    pairs = []
                    for j in sel:
                        flipped = base_vec.copy()
                        flipped[j] = 1 - flipped[j]
                        p_after = fe.predict_proba(flipped[None, :])[0]
                        score = float(np.max(np.abs(p_after - base_prob)))
                        pairs.append((j, score))
                    order[i] = np.asarray(
                        [j for (j, _) in sorted(pairs, key=lambda t: t[1], reverse=True)], dtype=int
                    )

                llm_cache = {
                    "mask_max": mask_max,
                    "C_true_llm": C_true_llm,
                    "C_before": C_before,
                    "y_prob_before": y_prob_before,
                    "order": order,
                }

            # derive current-budget mask from cached K=max selection
            mask_max = llm_cache["mask_max"]
            C_true_llm = llm_cache["C_true_llm"]
            C_before = llm_cache["C_before"]
            y_prob_before = llm_cache["y_prob_before"]
            order = llm_cache["order"]

            mask = np.zeros_like(mask_max, dtype=bool)
            for i in range(mask.shape[0]):
                if order[i].size:
                    k_take = int(min(budget, order[i].size))
                    if k_take > 0:
                        mask[i, order[i][:k_take]] = True

            overwrite_mask = mask & ~np.isnan(C_true_llm)
            C_after = np.where(overwrite_mask, C_true_llm, C_before)
            C_final_binary = (C_after >= 0.5).astype(int)
            y_prob_after = fe.predict_proba(C_final_binary)
            y_pred_after = np.argmax(y_prob_after, axis=1)

            result = SimpleNamespace(
                C_pred=C_before,
                C_intervened=C_after,
                mask=overwrite_mask,
                y_prob_before=y_prob_before,
                y_prob_after=y_prob_after,
                y_pred_after=y_pred_after,
            )

        else:
            # ── Standard (non-LLM) path ──
            result = runner.run(
                strategy=strategy,
                config=config,
                dataset=test,
                concept_proba=prob_test,
                labels=test.y.astype(int),
            )

            mask = result.mask
            C_gt = test.C.astype(np.float32)
            C_after = result.C_intervened.copy()

            mistake_draw = rng.random(C_after.shape) < err_prob
            mistakes = mask & mistake_draw
            C_after[mistakes] = 1.0 - C_gt[mistakes]
            result.C_intervened = C_after

            # Recompute downstream prediction after error injection
            C_final_binary = (result.C_intervened >= 0.5).astype(int)
            result.y_prob_after = fe.predict_proba(C_final_binary)
            result.y_pred_after = np.argmax(result.y_prob_after, axis=1)

        # Extract intervention statistics
        acc_intervened = float((result.y_pred_after == test.y.astype(int)).mean())

        n_intervened = int(np.sum(result.mask))
        n_samples = prob_test.shape[0]

        intervened_concepts = np.any(result.mask, axis=0)
        C_pred_binary = (result.C_pred >= 0.5).astype(int)
        C_final_binary = (result.C_intervened >= 0.5).astype(int)
        actual_edits_mask = C_pred_binary != C_final_binary
        prediction_num_concepts_intervened_on = {int(i): int(np.sum(actual_edits_mask[i])) for i in range(n_samples)}

        y_pred_before = np.argmax(result.y_prob_before, axis=1)
        num_preds_change = int(np.sum(result.y_pred_after != y_pred_before))

        concept_intervention_counts = {
            c: f"{int(np.sum(result.mask[:, i]))} ({int(np.sum(actual_edits_mask[:, i]))})"
            for i, c in enumerate(concept_names)
            if i < intervened_concepts.shape[0] and intervened_concepts[i]
        }

        key = f"top_{budget}_human_acc_{int(human_acc * 100)}"
        intervention_results[key] = {
            "accuracy": acc_intervened,
            "accuracy_gain": acc_intervened - acc_det,
            "predictions_intervened_on": int(np.sum(np.any(result.mask, axis=1))),
            "interventions_rate": float(np.sum(np.any(result.mask, axis=1)) / n_samples),
            "predictions_changed": num_preds_change,
            "avg_edits_per_intervention": float(
                sum(prediction_num_concepts_intervened_on.values())
            )
            / n_samples,
            "total_concept_confirmations": int(n_intervened),
            "total_concept_edits_made": int(sum(prediction_num_concepts_intervened_on.values())),
            "concept_interventions": concept_intervention_counts,
            "human_accuracy": human_acc,
        }

    return budgets, human_acc, intervention_results


# ── LLM/CLIP regime helper ────────────────────────────────────────────

def _run_llm_regime(config, regime, model, data, budgets, thresholds):
    """Run LLM or CLIP intervention regime.

    Trains LFCBM on the regime's concept descriptions, then calls
    ``_test_interventions`` with ``intervention_expert="llm"`` and
    ``FEOnProbs(lf.classifier)`` as the frontend. Matches the original
    ``automated_detection`` regime from ``robot_concept_regimes.py``.
    """
    from pathlib import Path

    _ensure_intervention_imports()
    from concept_benchmark.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig

    # Load concept descriptions for this regime
    from concept_benchmark.paths import pkg_dir
    if regime == "llm":
        concepts_file = config.llm_concepts_file
        if not concepts_file:
            concepts_file = str(pkg_dir / "concept_descriptions" / "llm.jsonl")
    else:  # clip
        concepts_file = config.clip_concepts_file
        if not concepts_file:
            concepts_file = str(pkg_dir / "concept_descriptions" / "clip.jsonl")

    p_cf = Path(concepts_file)
    if not p_cf.is_absolute():
        p_cf = (Path.cwd() / p_cf).resolve()
    if not p_cf.exists():
        raise FileNotFoundError(f"concepts file not found: {p_cf}")

    concept_set = LFConceptSet.from_file(str(p_cf))
    if not getattr(concept_set, "texts", None):
        raise ValueError(f"concepts file parsed empty: {p_cf}")

    # Train LFCBM on this regime's concepts (or load cached)
    device_str = str(determine_device())
    lfcbm_key = f"lfcbm_{regime}"
    lfcbm_path = config.get_model_path(lfcbm_key)

    if lfcbm_path.exists() and not config.force_retrain:
        logger.info("Loading existing LFCBM for %s: %s", regime, lfcbm_path)
        lf = load(lfcbm_path)
    else:
        cfg = LFTrainingConfig(
            device=device_str,
            seed=config.seed,
            cache_dir=config.get_model_path("lfcbm").parent / f"lfcbm_{regime}_cache",
        )
        lf = LabelFreeCBM(cfg)

        base_dir = getattr(data.training, "base_dir", Path("."))
        train_paths = [str((base_dir / Path(p)).resolve()) for p in data.training.X]
        valid_paths = [str((base_dir / Path(p)).resolve()) for p in data.validation.X]

        stats = lf.fit(
            train_X=train_paths,
            train_y=data.training.y.astype(int),
            valid_X=valid_paths,
            valid_y=data.validation.y.astype(int),
            concept_set=concept_set,
            cache_dir=cfg.cache_dir,
        )
        logger.info("LFCBM (%s) stats: %s/%s concepts kept", regime, stats.get("kept_concepts"), stats.get("total_concepts"))
        save(lf, lfcbm_path, overwrite=True)

    # Get concept probabilities from LFCBM
    base_dir = getattr(data.test, "base_dir", Path("."))
    test_paths = [str((base_dir / Path(p)).resolve()) for p in data.test.X]
    P_te = lf.concept_proba(test_paths)

    # Create FEOnProbs frontend (matching original)
    fe = FEOnProbs(lf.classifier)

    # Compute acc_det using continuous probs (matching original)
    y_pred_det = fe.predict_proba(P_te)
    acc_det = float((y_pred_det.argmax(1) == data.test.y.astype(int)).mean())

    # Matching original: human_annotation_accuracy = 0.8
    ia_val = config.expert_intervention_accuracy

    METRIC_COLS = [
        "accuracy",
        "predictions_intervened_on",
        "predictions_changed",
        "total_concept_confirmations",
        "total_concept_edits_made",
    ]

    COLS = ["budget", "threshold"] + METRIC_COLS
    all_dfs = []

    for t in thresholds:
        isettings = InterventionSettings(
            seed=config.seed,
            budgets=budgets,
            intervention_accuracy=ia_val,
            intervention_threshold=t,
            intervention_strategy=config.intervention_strategy,
            intervention_expert="llm",
            intervention_llm={
                "provider": config.llm_provider,
                "model": config.llm_model,
                "api_key": config.llm_api_key,
                "api_key_env": config.llm_api_key_env,
                "batch_size": 100,
            },
            run_dir=str(results_dir),
        )

        _, _, r = _test_interventions(
            prob_test=P_te,
            settings=isettings,
            acc_det=acc_det,
            fe=fe,
            test=data.test,
            concept_names=list(concept_set.keys),
        )
        df = (
            pd.DataFrame(r)
            .T.assign(budget=budgets)
            .assign(threshold=t)
            .reset_index(drop=True)[COLS]
        )
        all_dfs.append(df)

    regime_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    regime_df["regime"] = regime
    return regime_df


# ── Regime dispatch ───────────────────────────────────────────────────

def _run_regime(config, regime, model, data, budgets, thresholds):
    """Run one intervention regime. Returns list of result row dicts.

    ``model`` is always the *baseline* CBM (loaded once by the caller).
    For regimes that use a different CBM (e.g. "subjective"), this
    function loads the regime-specific model internally.
    """
    _ensure_intervention_imports()

    METRIC_COLS = [
        "accuracy",
        "predictions_intervened_on",
        "predictions_changed",
        "total_concept_confirmations",
        "total_concept_edits_made",
    ]

    # Select model, concept predictions, and human accuracy per regime
    c_preds = None  # set below; None means use regime_model.concept_detector
    regime_concept_names = None  # set for LFCBM regimes; None → use GT concepts
    if regime == "baseline":
        regime_model = model
        human_acc = config.intervention_accuracy
    elif regime == "expert":
        regime_model = model
        human_acc = config.expert_intervention_accuracy
    elif regime == "subjective":
        regime_model = load(config.get_model_path("cbm_subjective"))
        human_acc = config.subjective_intervention_accuracy
    elif regime == "machine":
        lfcbm_bundle = load(config.get_model_path("lfcbm"))
        lfcbm_obj = lfcbm_bundle["lfcbm"]
        fe_machine = lfcbm_bundle["frontend"]
        import os
        base = getattr(data.test, "base_dir", None)
        test_paths = [
            os.path.join(str(base), str(p)) if base else str(p)
            for p in data.test.X
        ]
        c_preds = lfcbm_obj.concept_proba(test_paths)
        regime_concept_names = list(lfcbm_obj.concept_set.keys)
        regime_model = ConceptBasedModel(concept_detector=None, front_end_model=fe_machine)
        human_acc = config.expert_intervention_accuracy
    elif regime in ("llm", "clip"):
        # LLM/CLIP regimes use separate concept files for corrections
        return _run_llm_regime(config, regime, model, data, budgets, thresholds)
    else:
        raise ValueError(f"Unknown regime: {regime!r}")

    if c_preds is None:
        c_preds = regime_model.concept_detector.predict(data.test)
    # For machine regime (FEOnProbs), pass continuous probs directly;
    # for other regimes, binarize first (matching original code).
    if regime == "machine":
        acc_det = float(
            (np.argmax(regime_model.front_end_model.predict_proba(c_preds),
                        axis=1) == data.test.y.astype(int)).mean()
        )
    else:
        acc_det = float(
            (np.argmax(regime_model.front_end_model.predict_proba(
                (c_preds >= 0.5).astype(int)), axis=1) == data.test.y.astype(int)).mean()
        )

    COLS = ["budget", "threshold"] + METRIC_COLS
    df_lst = []
    for t in thresholds:
        isettings = InterventionSettings(
            seed=config.seed,
            budgets=budgets,
            intervention_accuracy=human_acc,
            intervention_threshold=t,
            intervention_strategy=config.intervention_strategy,
        )
        _, _, r = _test_interventions(
            prob_test=c_preds,
            settings=isettings,
            acc_det=acc_det,
            fe=regime_model.front_end_model,
            test=data.test,
            concept_names=regime_concept_names,
        )
        df_lst.append(
            pd.DataFrame(r)
            .T.assign(budget=budgets)
            .assign(threshold=t)
            .reset_index(drop=True)[COLS]
        )

    regime_df = pd.concat(df_lst, axis=0).reset_index(drop=True)
    regime_df["regime"] = regime
    return regime_df


# ── Stage: run_interventions ──────────────────────────────────────────

def run_interventions(
    config: RobotBenchmarkConfig,
    model: Optional[ConceptBasedModel] = None,
    data=None,
) -> pd.DataFrame:
    """Run interventions on the trained CBM and return a results DataFrame.

    Loops over ``config.intervention_regimes`` (default: ``["baseline"]``).
    """
    _set_deterministic_seed(config.seed)
    _ensure_intervention_imports()
    patch_macos_dataloader()
    determine_device()

    if data is None:
        data = load(config.get_dataset_path())
    if model is None:
        model = load(config.get_model_path("cbm"))

    budgets = sorted(set(
        [0] + [data.n_concepts if b == -1 else b for b in config.intervention_budgets]
    ))
    thresholds = config.intervention_thresholds

    all_dfs = []
    for regime in config.intervention_regimes:
        try:
            regime_df = _run_regime(config, regime, model, data, budgets, thresholds)
            all_dfs.append(regime_df)
        except (FileNotFoundError, NotImplementedError) as e:
            logger.warning("Skipping regime %r: %s", regime, e)

    if not all_dfs:
        logger.warning("No regimes produced results.")
        return pd.DataFrame()

    results_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    results_df["data_name"] = "subconcept" if config.subconcept else "ideal"
    results_df["n"] = data.test.n
    results_df["concept_missing"] = config.concept_missing
    results_df["concept_missing_mech"] = config.concept_missing_mech
    results_df.to_csv(config.get_results_path("cbm"), index=False)
    return results_df


# ── Stage: align ─────────────────────────────────────────────────────

def align(
    config: RobotBenchmarkConfig,
    model: Optional[ConceptBasedModel] = None,
    data=None,
) -> dict:
    """Run alignment test on the trained CBM.

    Retrains the frontend with monotonicity (sign) constraints and
    compares original vs constrained accuracy.

    Returns dict with original_accuracy, aligned_accuracy, accuracy_change,
    predictions_changed, aligned_weights.
    """
    if data is None:
        data = load(config.get_dataset_path())
    if model is None:
        model = load(config.get_model_path("cbm"))

    return run_alignment(
        cbm=model,
        train_dataset=data.training,
        test_dataset=data.test,
        monotonicity_constraints=config.get_alignment_constraints(),
        save_path=config.get_alignment_results_path(),
    )


# ── Stage: collect_results ────────────────────────────────────────────

def _dataset_label(cfg: RobotBenchmarkConfig) -> str:
    """Return a human-readable dataset label for a config."""
    base = "subconcept" if cfg.subconcept else "ideal"
    if cfg.concept_missing_mech != "none":
        return f"{base}_{cfg.concept_missing_mech}"
    return base


def collect_results(
    configs: Optional[List[RobotBenchmarkConfig]] = None,
) -> pd.DataFrame:
    """Aggregate all robot results into a single flat CSV.

    Produces one row per (dataset, model, budget) combination with columns:
      dataset, model, budget, threshold, accuracy, gain,
      predictions_intervened_on, avg_concepts_per_sample, predictions_changed

    Reads saved artifacts only — no model retraining.
    """
    import json

    if configs is None:
        configs = [RobotBenchmarkConfig.default_ideal()]

    device = determine_device()
    loader_config = get_loader_config(device)
    rows = []

    # ── Per-config: DNN, CBM, interventions, alignment ───────────────
    for cfg in configs:
        label = _dataset_label(cfg)

        # Load this config's dataset
        data = load(cfg.get_dataset_path())

        # DNN accuracy
        dnn_path = cfg.get_model_path("dnn")
        dnn_accuracy = None
        if dnn_path.exists():
            dnn_weights = load(dnn_path)
            dnn = RobotClassifierCNN(input_size=cfg.input_size).to(device)
            dnn.load_state_dict(dnn_weights)
            test_loader = data.test.loader(shuffle=False, **loader_config)
            dnn_accuracy = compute_accuracy(dnn, test_loader, device)
            rows.append({
                "dataset": label,
                "model": "dnn",
                "budget": "",
                "threshold": "",
                "accuracy": round(dnn_accuracy, 4),
                "gain": 0.0,
                "predictions_intervened_on": "",
                "avg_concepts_per_sample": "",
                "predictions_changed": "",
            })

        # CBM no-intervention (k=0)
        cbm_path = cfg.get_model_path("cbm")
        if not cbm_path.exists():
            logger.warning("CBM not found for %s, skipping: %s", label, cbm_path)
            continue
        cbm = load(cbm_path)
        cbm_acc = float((cbm.predict(data.test) == data.test.y).mean())
        gain_ref = dnn_accuracy if dnn_accuracy is not None else cbm_acc
        rows.append({
            "dataset": label,
            "model": "cbm",
            "budget": 0,
            "threshold": "",
            "accuracy": round(cbm_acc, 4),
            "gain": round(cbm_acc - gain_ref, 4),
            "predictions_intervened_on": "",
            "avg_concepts_per_sample": "",
            "predictions_changed": "",
        })

        # CBM with interventions (k>0)
        results_path = cfg.get_results_path("cbm")
        if results_path.exists():
            interv_df = pd.read_csv(results_path)
            # Filter to baseline regime if column present
            if "regime" in interv_df.columns:
                interv_df = interv_df[interv_df["regime"] == "baseline"]
            # Use threshold=0.2 as the canonical threshold for the summary
            t02 = interv_df[(interv_df["threshold"] == 0.2) & (interv_df["budget"] > 0)]
            for _, row in t02.iterrows():
                budget = int(row["budget"])
                acc = float(row["accuracy"])
                pio = int(row["predictions_intervened_on"])
                tcc = int(row["total_concept_confirmations"])
                avg_cps = round(tcc / pio, 2) if pio > 0 else 0.0
                rows.append({
                    "dataset": label,
                    "model": "cbm",
                    "budget": budget,
                    "threshold": 0.2,
                    "accuracy": round(acc, 4),
                    "gain": round(acc - gain_ref, 4),
                    "predictions_intervened_on": pio,
                    "avg_concepts_per_sample": avg_cps,
                    "predictions_changed": int(row["predictions_changed"]),
                })

        # Aligned CBM (only for non-missingness configs)
        if cfg.concept_missing_mech == "none":
            align_path = cfg.get_alignment_results_path()
            if align_path.exists():
                with open(align_path) as f:
                    align_data = json.load(f)
                aligned_acc = float(align_data["aligned_accuracy"])
                rows.append({
                    "dataset": label,
                    "model": "aligned_cbm",
                    "budget": 0,
                    "threshold": "",
                    "accuracy": round(aligned_acc, 4),
                    "gain": round(aligned_acc - gain_ref, 4),
                    "predictions_intervened_on": "",
                    "avg_concepts_per_sample": "",
                    "predictions_changed": "",
                })

                # Aligned CBM with intervention at k=3
                aligned_weights = align_data.get("aligned_weights")
                if aligned_weights is not None:
                    from concept_benchmark.alignment import align_frontend_weights
                    import copy as _copy

                    # Load the config's own dataset so concept shapes match
                    cfg_data = load(cfg.get_dataset_path())
                    aligned_fe = _copy.deepcopy(cbm.front_end_model)
                    aligned_fe = align_frontend_weights(
                        aligned_fe, list(cfg_data.test.concepts), aligned_weights,
                    )
                    c_preds = cbm.concept_detector.predict(cfg_data.test)
                    isettings = InterventionSettings(
                        seed=cfg.seed,
                        budgets=[3],
                        intervention_accuracy=cfg.intervention_accuracy,
                        intervention_threshold=0.2,
                    )
                    _, _, int_results = _test_interventions(
                        prob_test=c_preds,
                        settings=isettings,
                        acc_det=aligned_acc,
                        fe=aligned_fe,
                        test=cfg_data.test,
                    )
                    for key, res in int_results.items():
                        pio = int(res["predictions_intervened_on"])
                        tcc = int(res["total_concept_confirmations"])
                        avg_cps = round(tcc / pio, 2) if pio > 0 else 0.0
                        rows.append({
                            "dataset": label,
                            "model": "aligned_cbm",
                            "budget": 3,
                            "threshold": 0.2,
                            "accuracy": round(float(res["accuracy"]), 4),
                            "gain": round(float(res["accuracy"]) - gain_ref, 4),
                            "predictions_intervened_on": pio,
                            "avg_concepts_per_sample": avg_cps,
                            "predictions_changed": int(res["predictions_changed"]),
                        })

    final_df = pd.DataFrame(rows)
    cfg0 = configs[0]
    out_path = cfg0.get_collect_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(final_df), out_path)
    return final_df


# ── Stage: run (orchestrator) ─────────────────────────────────────────

def run(
    config: Optional[RobotBenchmarkConfig] = None,
    stages: Optional[List[str]] = None,
    force_setup: bool = False,
) -> None:
    """Run the robot benchmark pipeline for a single configuration.

    Args:
        config: Benchmark configuration. Defaults to ideal.
        stages: List of stages to run. Default: all.
        force_setup: If True, delete cached images/data before regenerating.
    """
    from concept_benchmark._logging import setup_logging
    setup_logging()
    patch_macos_dataloader()

    if config is None:
        config = RobotBenchmarkConfig.default_ideal()
    if stages is None:
        stages = ["setup", "cbm", "dnn", "intervene", "align", "collect"]

    # Early validation: check that dataset exists if we need it
    _needs_data = {"cbm", "dnn", "intervene", "align", "collect"}
    if _needs_data & set(stages) and "setup" not in stages:
        ds_path = config.get_dataset_path()
        if not ds_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {ds_path}\n"
                f"Run with --stages setup first, or include 'setup' in --stages."
            )

    device = determine_device()
    variant = "subconcept" if config.subconcept else "ideal"
    n_stages = len(stages)
    _si = {s: i for i, s in enumerate(stages, 1)}
    logger.info(
        "=== Robot Benchmark === seed=%d, variant=%s, stages=%s, device=%s",
        config.seed, variant, stages, device,
    )

    if "setup" in stages:
        logger.info("=== [%d/%d] Setup ===", _si["setup"], n_stages)
        import shutil
        fp_path = config.get_dataset_path().with_suffix(".fingerprint")
        current_fp = config.setup_fingerprint()
        cached_fp = fp_path.read_text().strip() if fp_path.exists() else None

        if force_setup or cached_fp != current_fp:
            if force_setup:
                logger.info("--force-setup: regenerating data from scratch")
            elif cached_fp is None:
                logger.info("No cached data found — generating dataset and robot images (this may take a minute)")
            else:
                logger.info("Config changed since last setup — regenerating data")
            # Clear cached images and dataset
            img_dir = config.to_dict()["output_directory"]
            if Path(img_dir).exists():
                shutil.rmtree(img_dir)
            ds_path = config.get_dataset_path()
            if ds_path.exists():
                ds_path.unlink()
            setup_dataset(config)
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            fp_path.write_text(current_fp)
        else:
            logger.info("Setup data is up to date (fingerprint matches), skipping")

    # Model fingerprint: retrain if config changed since last training
    model_fp_path = config.get_model_path("cbm").with_suffix(".fingerprint")
    current_model_fp = config.model_fingerprint()
    cached_model_fp = model_fp_path.read_text().strip() if model_fp_path.exists() else None
    model_stale = cached_model_fp != current_model_fp

    def _should_train(model_key: str) -> bool:
        model_path = config.get_model_path(model_key)
        if config.force_retrain:
            return True
        if not model_path.exists():
            return True
        if model_stale:
            logger.info("Config changed since last training — retraining %s", model_key)
            return True
        return False

    if "cbm" in stages:
        logger.info("=== [%d/%d] Train CBM ===", _si["cbm"], n_stages)
        if _should_train("cbm"):
            train_cbm(config)
        else:
            logger.info("Using existing CBM: %s", config.get_model_path("cbm"))
        if "subjective" in config.intervention_regimes:
            if _should_train("cbm_subjective"):
                train_cbm_subjective(config)
            else:
                logger.info("Using existing subjective CBM: %s", config.get_model_path("cbm_subjective"))
        if "machine" in config.intervention_regimes:
            if _should_train("lfcbm"):
                train_lfcbm(config)
            else:
                logger.info("Using existing LFCBM: %s", config.get_model_path("lfcbm"))

    if "dnn" in stages:
        logger.info("=== [%d/%d] Train DNN ===", _si["dnn"], n_stages)
        if _should_train("dnn"):
            train_dnn(config)
        else:
            logger.info("Using existing DNN: %s", config.get_model_path("dnn"))

    # Save model fingerprint after training stages
    if ("cbm" in stages or "dnn" in stages) and model_stale:
        model_fp_path.parent.mkdir(parents=True, exist_ok=True)
        model_fp_path.write_text(current_model_fp)

    if "intervene" in stages:
        logger.info("=== [%d/%d] Intervene ===", _si["intervene"], n_stages)
        run_interventions(config)

    if "align" in stages:
        logger.info("=== [%d/%d] Align ===", _si["align"], n_stages)
        align(config)

    if "collect" in stages:
        logger.info("=== [%d/%d] Collect ===", _si["collect"], n_stages)
        collect_results([config])


# ── CLI entry point ──────────────────────────────────────────────────

ROBOT_STAGES = ("setup", "cbm", "dnn", "intervene", "align", "collect")


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the robot classification benchmark pipeline.",
    )
    parser.add_argument("--seed", type=int, default=1014)
    parser.add_argument(
        "--stages", nargs="+", default=list(ROBOT_STAGES),
        help=f"Pipeline stages to run (default: all). Valid: {' -> '.join(ROBOT_STAGES)}",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--subconcept", action="store_true")
    parser.add_argument("--concept-missing", type=float, default=None,
                        help="Fraction of concept labels to mask (e.g. 0.2).")
    parser.add_argument("--concept-missing-mech", type=str, default=None,
                        choices=["none", "mcar", "mnar"])
    parser.add_argument("--budgets", nargs="+", default=None,
                        help="Intervention budgets (e.g. 1 3 5 max).")
    parser.add_argument("--regimes", nargs="+", default=None,
                        help="Intervention regimes (e.g. baseline expert subjective machine).")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["kflip", "exact_k"])
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-setup", action="store_true")
    return parser.parse_args(argv)


def _parse_budgets(raw):
    budgets = []
    for v in raw:
        budgets.append(-1 if v.lower() == "max" else int(v))
    return budgets


def main(argv=None):
    args = _parse_args(argv)

    unknown = set(args.stages) - set(ROBOT_STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}. Valid: {list(ROBOT_STAGES)}")

    if args.config:
        config = RobotBenchmarkConfig.from_yaml(args.config)
    elif args.subconcept:
        config = RobotBenchmarkConfig.default_subconcept()
        config.seed = args.seed
    else:
        config = RobotBenchmarkConfig(seed=args.seed)

    if args.budgets:
        config.intervention_budgets = _parse_budgets(args.budgets)
    if args.regimes:
        config.intervention_regimes = args.regimes
    if args.strategy:
        config.intervention_strategy = args.strategy
    if args.llm_api_key:
        config.llm_api_key = args.llm_api_key
    if args.force_retrain:
        config.force_retrain = True
    if args.concept_missing is not None:
        config.concept_missing = args.concept_missing
        config.concept_missing_mech = args.concept_missing_mech or "mcar"
    elif args.concept_missing_mech is not None:
        config.concept_missing_mech = args.concept_missing_mech

    run(config, stages=args.stages, force_setup=args.force_setup)


if __name__ == "__main__":
    main()
