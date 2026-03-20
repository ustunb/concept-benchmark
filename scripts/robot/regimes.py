"""Robot pipeline — intervention regime dispatch and helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from concept_benchmark.utils import determine_device
from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.paths import data_dir, results_dir
from experiments.models import ConceptBasedModel

from .training import FEOnProbs

logger = logging.getLogger(__name__)


@dataclass
class InterventionSettings:
    """Typed config for _test_interventions, replacing the old ``sttngs`` dict."""

    seed: int
    budgets: list[int]
    intervention_accuracy: float = 0.9
    intervention_threshold: float = 1.0
    intervention_strategy: str = "up_to_k"
    intervention_expert: str = ""  # "" for standard path, "llm" for inline LLM
    intervention_llm: dict[str, Any] | None = None
    run_dir: str = "."


# Lazy import to avoid circular deps — intervention modules
_intervention_imported = False


def _ensure_intervention_imports():
    global _intervention_imported
    if not _intervention_imported:
        global ConceptInterventionRunner, InterventionConfig, KFlipInterventionStrategy
        from experiments.intervention import (
            ConceptInterventionRunner,
            InterventionConfig,
        )
        from experiments.kflip import KFlipInterventionStrategy

        _intervention_imported = True


def _test_interventions(prob_test, settings: InterventionSettings, acc_det, fe, test, concept_names=None):
    """Run interventions for each budget and return results dict.

    Matches the original ``test_interventions`` from ``robot_concept_regimes.py``,
    including the inline LLM path (compute-once at K=max, batched multi-image calls,
    JSONL cache, flip-effect ranking, per-budget mask derivation).
    """
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
                logger.warning("Truncating concept_proba from %d to %d columns to match GT shape", n_pred, n_gt)
                prob_test = prob_test[:, :n_gt]
            else:
                logger.warning("Padding concept_proba from %d to %d columns to match GT shape", n_pred, n_gt)
                pad = np.zeros((prob_test.shape[0], n_gt - n_pred), dtype=prob_test.dtype)
                prob_test = np.concatenate([prob_test, pad], axis=1)
        if len(concept_names) != prob_test.shape[1]:
            concept_names = concept_names[: prob_test.shape[1]]

    # Create a CBM wrapper for the intervention framework
    cbm = ConceptBasedModel(concept_detector=None, label_predictor=fe)
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
            intervention_noise_rate=1.0 - human_acc,
        )

        strategy = KFlipInterventionStrategy(
            use_exact_k=(settings.intervention_strategy == "exactly_k"),
        )

        if settings.intervention_expert.lower() == "llm":
            # ── Inline LLM path (matching original robot_concept_regimes.py) ──
            try:
                from google.api_core.exceptions import ResourceExhausted
            except ImportError:
                ResourceExhausted = None

            from experiments.llm_client import make_llm_client

            llm_cfg = settings.intervention_llm or {}
            provider = str(llm_cfg.get("provider", "gemini"))
            model_name = str(llm_cfg.get("model", "gemini-2.5-flash-lite"))
            api_key_env = str(llm_cfg.get("api_key_env", "GEMINI_API_KEY"))

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
                if p.is_absolute():
                    return str(p)
                return str((data_dir / "robot_images" / p).resolve())

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
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning("JSON parse failure for LLM response: %s (raw=%r)", e, raw[:200] if raw else raw)
                if not parsed:
                    logger.warning("LLM judge returned empty dict for image=%s", image_path)
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
                    intervention_noise_rate=1.0 - human_acc,
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
                            for line_num, line in enumerate(f, 1):
                                try:
                                    rec = json.loads(line)
                                    i0 = int(rec["i"])
                                    votes_idx = {int(k): int(v) for k, v in rec.get("votes_idx", {}).items()}
                                    d[i0] = votes_idx
                                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                                    logger.warning("Skipping malformed cache line %d in %s: %s", line_num, cache_path, e)
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
                order = [np.array([], dtype=int) for _ in range(C_before.shape[0])]
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
    _ensure_intervention_imports()
    from experiments.lfcbm import LabelFreeCBM, LFConceptSet, LFTrainingConfig

    # Load concept descriptions for this regime
    from concept_benchmark.paths import package_dir
    if regime == "llm":
        concepts_file = config.llm_concepts_file
        if not concepts_file:
            concepts_file = str(package_dir / "concept_descriptions" / "llm.jsonl")
    else:  # clip
        concepts_file = config.clip_concepts_file
        if not concepts_file:
            concepts_file = str(package_dir / "concept_descriptions" / "clip.jsonl")

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

        image_dir = data_dir / "robot_images"
        train_paths = [str(image_dir / p) for p in data.training.X]
        valid_paths = [str(image_dir / p) for p in data.validation.X]

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
    image_dir = data_dir / "robot_images"
    test_paths = [str(image_dir / p) for p in data.test.X]
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
        image_dir = data_dir / "robot_images"
        test_paths = [str(image_dir / p) for p in data.test.X]
        c_preds = lfcbm_obj.concept_proba(test_paths)
        regime_concept_names = list(lfcbm_obj.concept_set.keys)
        regime_model = ConceptBasedModel(concept_detector=None, label_predictor=fe_machine)
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
            (np.argmax(regime_model.label_predictor.predict_proba(c_preds),
                        axis=1) == data.test.y.astype(int)).mean()
        )
    else:
        acc_det = float(
            (np.argmax(regime_model.label_predictor.predict_proba(
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
            fe=regime_model.label_predictor,
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
