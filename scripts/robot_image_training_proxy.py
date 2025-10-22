import os, json, time, pickle, getpass, pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import copy
from torchvision import transforms

# Headless Qt defaults
os.environ.setdefault("XDG_RUNTIME_DIR", f"/tmp/runtime-{getpass.getuser()}")
Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
os.chmod(os.environ["XDG_RUNTIME_DIR"], 0o700)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from concept_benchmark.models import ConceptDetector, FrontEndModel, RobotConceptClassifier
from concept_benchmark.paths import results_dir
from scripts.dataset_skewing import create_skewed_splits, filter_training_by_string
from scripts.dnn_training import train_eval_image
from scripts.interventions import apply_interventions
from concept_benchmark.synthetic.proxy import create_synthetic_dataset

try:
    import psutil as _psutil
except Exception:
    _psutil = None
from torchvision import transforms

def _mem(tag):
    try:
        rss = _psutil.Process(os.getpid()).memory_info().rss / (1024**3) if _psutil else -1.0
        print(f"[MEM] {tag} rss_gb={rss:.3f}", flush=True)
    except Exception:
        print(f"[MEM] {tag} rss_gb=N/A", flush=True)

settings = {
    "samples_per_instance": 1,
    "draw": 0,
    "CBM_type": "joint",  # or "sequential"
    "image_dir": "./data/robot_images",
    "image_size": "medium",
    "color_mode": "color",
    "train_dnn": 0,
    "seed": 555,
    # Oracle label depends ONLY on coarse foot and coarse hand
    "model": "'glorp' if (int(str(row['foot_shape']).startswith('pointy')) + int(str(row['hand_shape']).startswith('edgy')))>= 2 else 'drent'",
    "dataset_characterization": "",
    "knows_concepts": False,  # False => expose subtypes; True => coarse only
    "spurious_features": ["has_elbows", "hand_shape_subtype", "foot_shape_subtype"],
    "drop_concepts": [],
    "human_alignment": {"foot_shape": 1, "hand_shape": 1, "bias": -0.01},
    "model_type": "stochastic",
    "logit_scalar": 2.0,
    "logit_intercept": 2.5,
    "label_noise_rate": 0.0,
    "missingness": "complete",
    "missing_rate": 1.0,
    "impute_missing": 0,
    # Default skew targets a few subtypes from BOTH NS features
    "skew_concept": [
        {"concepts": {"foot_shape_pointy_3sided": 1}, "min_fraction": 0.20},
        {"concepts": {"foot_shape_pointy_4sided": 1}, "min_fraction": 0.20},
        {"concepts": {"hand_shape_edgy_square": 1}, "min_fraction": 0.20},
        {"concepts": {"hand_shape_edgy_trapezoid": 1}, "min_fraction": 0.20},
    ],
    "budget": [1, 10],
    "intervention_accuracy": 0.9,
    "intervention_threshold": 0.1,
    "epochs": 10,
    "out_dir": str(results_dir / "robots"),
    "run_name": "proxy_setup_baseline",
    "load_detector": "",
    "load_frontend": "",
    # Proxy spec: body_shape proxies foot_shape; ears_shape proxies hand_shape
    "proxy_spec": {
        "body_shape": {
            "source": "foot_shape",
            "p": 0.7,
            "source_to_bit": {"flat_trapezoid": 0, "flat_rounded": 0, "flat_square": 0, "flat_5sided": 0, "flat_lshaped": 0,
                              "pointy_trapezoid": 1, "pointy_rounded": 1, "pointy_square": 1, "pointy_3sided": 1, "pointy_4sided": 1},
            "bit_to_value": {0: "square", 1: "round"}
        },
        "ears_shape": {
            "source": "hand_shape",
            "p": 0.6,
            "source_to_bit": {"round_circle": 0, "round_oval": 0, "round_oval2": 0,
                              "edgy_triangle": 1, "edgy_square": 1, "edgy_trapezoid": 1},
            "bit_to_value": {0: "square", 1: "triangle"}
        }
    },
    # Mode: "oracle" => drop proxies and subtypes; "real" => drop coarse NS, keep proxies
    "mode": "real",
    "fe_harness": 1,

    # New controls (reverse compatible defaults)
    "enforce_pattern_limits": 0,
    "max_feet_per_pattern": 2,
    "max_exposures_per_pattern": 2,
    "disjoint_patterns_across_splits": 0,
    "target_train_unique": 0,
    "coarse_balance_feature": "",
    "proxy_p": None,
    "subtype_label_bias": {},
}

def _rate_tag(r):
    v = int(round(float(r) * 100))
    return f"{v:03d}"

def _apply_label_noise(sample, noise_rate, seed):
    if noise_rate <= 0:
        return sample
    rng = np.random.default_rng(int(seed) + 4242)
    y = sample.y.astype(int).copy()
    flip_mask = rng.random(y.shape[0]) < float(noise_rate)
    y[flip_mask] = 1 - y[flip_mask]
    return sample.__class__(
        parent=sample.parent, X=sample.X, C=sample.C, y=y, meta=sample.meta,
        transform=sample.transform, concept_transform=sample.concept_transform,
        target_transform=sample.target_transform, base_dir=getattr(sample, 'base_dir', None)
    )

def _get_foot_shape_pred(pred_row, concept_names):
    if 'foot_shape' in concept_names:
        return int(pred_row[concept_names.index('foot_shape')])
    pointy_types = [c for c in concept_names if 'foot_shape_pointy' in c]
    for ptype in pointy_types:
        if pred_row[concept_names.index(ptype)] == 1:
            return 1
    return 0

def _dedupe_train_by_robot_ids(train):
    ids_full = np.asarray(train.meta.get("robot_ids"))
    if ids_full is None or ids_full.size == 0:
        return train

    df_idx = train.meta.get("df_indices")
    if df_idx is None:
        if ids_full.shape[0] != len(train):
            raise ValueError("robot_ids misaligned with train and df_indices missing")
        ids = ids_full
    else:
        idx = np.asarray(df_idx)
        if idx.dtype.kind == "b":
            if idx.size != ids_full.shape[0]:
                raise ValueError("boolean df_indices length mismatch")
            idx = np.flatnonzero(idx)
        elif idx.dtype.kind not in "iu":
            if not np.all(np.isfinite(idx)):
                raise ValueError("df_indices contains NaN/inf")
            if not np.allclose(idx, np.floor(idx)):
                raise ValueError("df_indices contains non-integers")
            idx = idx.astype(np.int64)
        else:
            idx = idx.astype(np.int64)

        if idx.min() < 0 or idx.max() >= ids_full.shape[0]:
            raise ValueError("df_indices out of bounds for robot_ids")
        ids = ids_full[idx]

    keep = np.unique(ids, return_index=True)[1]
    m = np.zeros(len(train), dtype=bool)
    m[keep] = True
    return train.filter(m)

def _enforce_pattern_limits(train, data, S):
    if not int(S.get("enforce_pattern_limits", 0)):
        return train
    cat = data.meta.get("catalog_df")
    if cat is None:
        return train
    foot_col = "foot_shape_subtype" if "foot_shape_subtype" in cat.columns else "foot_shape"
    patt_cols = [c for c in ["head_shape","body_shape","has_antennae","mouth_type"] if c in cat.columns]
    if not patt_cols or foot_col not in cat.columns:
        return train
    ids_tr = train.meta.get("robot_ids")
    df_tr = cat.set_index("id").loc[ids_tr].reset_index(drop=True)
    df_tr["_p"] = df_tr[patt_cols].astype(str).agg("|".join, axis=1)
    max_feet = int(S.get("max_feet_per_pattern", 2))
    max_exp = int(S.get("max_exposures_per_pattern", 2))
    rng_np = np.random.default_rng(int(S["seed"]))
    keep_idx = []
    for _, g in df_tr.groupby("_p"):
        feet = list(g[foot_col].dropna().unique())
        rng_np.shuffle(feet)
        feet = feet[:max_feet]
        g2 = g[g[foot_col].isin(feet)]
        picks = []
        for f in feet:
            idxs = g2.index[g2[foot_col]==f].to_numpy()
            if len(idxs)>0:
                picks.append(rng_np.choice(idxs))
        if len(picks) < max_exp:
            remaining = np.setdiff1d(g2.index.to_numpy(), np.array(picks))
            if len(remaining)>0:
                add = rng_np.choice(remaining, size=min(max_exp-len(picks), len(remaining)), replace=False)
                picks = np.concatenate([np.atleast_1d(picks), np.atleast_1d(add)])
        keep_idx.extend([int(i) for i in np.atleast_1d(picks)])
    if keep_idx:
        mask = np.zeros(len(df_tr), dtype=bool)
        mask[np.array(keep_idx)] = True
        train = train.filter(mask)
    return train

def _balance_and_cap_train(train, data, S):
    target = int(S.get("target_train_unique", 0))
    coarse = str(S.get("coarse_balance_feature", "")).strip()
    if target <= 0 and not coarse:
        return train
    cat = data.meta.get("catalog_df")
    if cat is None:
        return train
    ids = train.meta.get("robot_ids")
    df_tr = cat.set_index("id").loc[ids].reset_index(drop=True)
    keep_idx = np.arange(len(df_tr))
    if coarse:
        if coarse not in df_tr.columns:
            if coarse == "foot_shape" and "foot_shape_subtype" in df_tr.columns and "foot_shape" in df_tr.columns:
                pass
            else:
                coarse = ""
        if coarse:
            grp_vals = df_tr[coarse].astype(str).values
            uniq = np.unique(grp_vals)
            if target <= 0:
                counts = {u: (grp_vals==u).sum() for u in uniq}
                n = min(counts.values())
                rng = np.random.default_rng(int(S["seed"]))
                keep = []
                for u in uniq:
                    idxs = np.where(grp_vals==u)[0]
                    if len(idxs)>n:
                        idxs = rng.choice(idxs, size=n, replace=False)
                    keep.append(idxs)
                keep_idx = np.concatenate(keep)
            else:
                per = max(1, target // max(1, len(uniq)))
                rng = np.random.default_rng(int(S["seed"]))
                keep = []
                for u in uniq:
                    idxs = np.where(grp_vals==u)[0]
                    k = min(len(idxs), per)
                    sel = rng.choice(idxs, size=k, replace=False)
                    keep.append(sel)
                keep_idx = np.concatenate(keep)
    if target > 0 and not coarse:
        rng = np.random.default_rng(int(S["seed"]))
        if len(keep_idx) > target:
            keep_idx = rng.choice(keep_idx, size=target, replace=False)
    keep_mask = np.zeros(len(df_tr), dtype=bool)
    keep_mask[np.array(keep_idx, dtype=int)] = True
    train = train.filter(keep_mask)
    return train

def _final_hard_cap_train(train, data, S):
    target = int(S.get("target_train_unique", 0))
    if target <= 0 or len(train) <= target:
        return train
    rng = np.random.default_rng(int(S["seed"]))
    coarse = str(S.get("coarse_balance_feature", "")).strip()
    groups = None
    if coarse:
        cat = data.meta.get("catalog_df")
        if cat is not None and "id" in cat.columns and coarse in cat.columns:
            ids_full = np.asarray(data.meta.get("robot_ids"))
            df_idx = train.meta.get("df_indices")
            ids_tr = None
            if df_idx is not None:
                idx = np.asarray(df_idx)
                if idx.dtype.kind == "b":
                    idx = np.flatnonzero(idx)
                elif idx.dtype.kind not in "iu":
                    idx = idx.astype(np.int64)
                else:
                    idx = idx.astype(np.int64)
                if idx.size > 0 and idx.max() < ids_full.shape[0]:
                    ids_tr = ids_full[idx]
            else:
                rid = train.meta.get("robot_ids")
                if rid is not None and len(rid) == len(train):
                    ids_tr = np.asarray(rid)
            if ids_tr is not None:
                try:
                    groups = cat.set_index("id").loc[ids_tr][coarse].astype(str).values
                except Exception:
                    groups = None
    n = len(train)
    if groups is None:
        sel = rng.choice(np.arange(n, dtype=int), size=target, replace=False)
    else:
        uniq = np.unique(groups)
        sizes = np.array([(groups == u).sum() for u in uniq], dtype=int)
        prop = target * sizes / float(n)
        base = np.floor(prop).astype(int)
        rem = target - base.sum()
        frac = prop - base
        order = rng.permutation(len(uniq))
        take_more = np.argsort(frac[order])[::-1][:rem]
        base[order[take_more]] += 1
        picks = []
        pool_all = np.arange(n, dtype=int)
        for u, k in zip(uniq, base):
            idxs = np.where(groups == u)[0]
            if k > 0 and idxs.size > 0:
                k = min(k, idxs.size)
                picks.append(rng.choice(idxs, size=k, replace=False))
        sel = np.concatenate(picks) if picks else np.array([], dtype=int)
        if sel.size < target:
            remaining = np.setdiff1d(pool_all, sel, assume_unique=False)
            if remaining.size > 0:
                topup = rng.choice(remaining, size=min(target - sel.size, remaining.size), replace=False)
                sel = np.concatenate([sel, topup])
    m = np.zeros(n, dtype=bool)
    m[sel[:target]] = True
    return train.filter(m)
    
def main(sttngs):
    S = dict(sttngs)
    rng = np.random.default_rng(int(S["seed"]))
    base_out = Path(S["out_dir"]); base_out.mkdir(parents=True, exist_ok=True)
    miss = str(S["missingness"]).lower()
    rate = float(S["missing_rate"])
    int_acc_tag = f"int-acc{int(round(float(S['intervention_accuracy']) * 100))}"
    miss_tag = "complete" if miss == "complete" or rate <= 0 else f"{miss}{_rate_tag(rate)}"
    skew_tag = f"_skew" if S.get("skew_concept", []) != [] else ""
    impute_tag = f"impute{int(S['impute_missing'])}"
    filter_tag = "_filter" if S.get("dataset_characterization", "") != "" else ""
    label_noise_tag = "_label-noise_" if float(S.get("label_noise_rate", 0.0)) != 0.0 else "_"
    seed_tag = f"seed{int(S['seed'])}"
    model_type_tag = f"{S['model_type']}_"
    slug = f"robots_proxy_{model_type_tag}{miss_tag}{filter_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{impute_tag}"
    run_dir = base_out / (S["run_name"] or f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"); run_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([transforms.ToTensor(),])

    proxy_spec = copy.deepcopy(S.get("proxy_spec", {}))
    if S.get("proxy_p") is not None:
        try:
            pval = float(S["proxy_p"])
            for k in proxy_spec:
                proxy_spec[k]["p"] = pval
        except Exception:
            pass

    params = {
        "samples_per_instance": int(S["samples_per_instance"]),
        "draw": bool(int(S["draw"])),
        "output_directory": S.get("image_dir", run_dir / "images"),
        "concepts": {
            "head_shape": ["square", "round"],
            "body_shape": ["square", "round"],
            "has_knees": ["false", "true"],
            "has_elbows": ["false", "true"],
            "has_antennae": ["false", "true"],
            "ears_shape": ["square", "triangle"],
            "mouth_type": ["closed", "open"],
            "hand_shape": [
                "round_circle", "round_oval", "round_oval2",
                "edgy_triangle", "edgy_square", "edgy_trapezoid",
            ],
            "foot_shape": [
                "flat_trapezoid", "flat_rounded", "flat_square", "flat_5sided", "flat_lshaped",
                "pointy_trapezoid", "pointy_rounded", "pointy_square", "pointy_3sided", "pointy_4sided",
            ],
        },
        "spurious_features": S.get("spurious_features", ["has_elbows"]),
        "model": S.get("model"),
        "model_type": S["model_type"],
        "scalar": float(S.get("logit_scalar", 1.0)),
        "intercept": float(S.get("logit_intercept", 0.0)),
        "size": S["image_size"],
        "color_mode": str(S["color_mode"]),
        "test_set_size": 10000,
        "train_concept_detector": True,
        "epochs": int(S["epochs"]),
        "verbose": True,
        "rng_seed": S['seed'],
        "proxy_spec": proxy_spec,
        "proxy_p": S.get("proxy_p", None),
        "subtype_label_bias": S.get("subtype_label_bias", {}),
    }

    data = create_synthetic_dataset(**params)
    data.transform = tf
    data.generate_cvindices(seed=int(S["seed"]))

    mode = str(S.get("mode", "real")).lower()
    if mode == "oracle":
        drop_list = ["body_shape", "ears_shape",
                     "foot_shape_flat_trapezoid","foot_shape_flat_rounded","foot_shape_flat_square","foot_shape_flat_5sided","foot_shape_flat_lshaped",
                     "foot_shape_pointy_trapezoid","foot_shape_pointy_rounded","foot_shape_pointy_square","foot_shape_pointy_3sided","foot_shape_pointy_4sided",
                     "hand_shape_round_circle","hand_shape_round_oval","hand_shape_round_oval2",
                     "hand_shape_edgy_triangle","hand_shape_edgy_square","hand_shape_edgy_trapezoid"]
    else:
        drop_list = ["foot_shape","hand_shape"]

    if S.get("skew_concept"):
        try:
            train, valid, test = create_skewed_splits(data, skew_specs=S["skew_concept"], rng=rng, drop_concepts=drop_list, fractions_unique=True)
        except TypeError:
            train, valid, test = create_skewed_splits(data, skew_specs=S["skew_concept"], rng=rng, drop_concepts=drop_list)
    elif S.get("dataset_characterization", "") != "":
        train, valid, test = filter_training_by_string(data, string=S["dataset_characterization"], rng=rng)
    else:
        data.drop_concepts(drop_list)
        data.split("K05N01", fold_num_validation=4, fold_num_test=5)
        train = data.training; valid = data.validation; test = data.test

    if int(S.get("disjoint_patterns_across_splits", 0)):
        cat = data.meta.get("catalog_df")
        if cat is not None:
            patt_cols = [c for c in ["head_shape","body_shape","has_antennae","mouth_type"] if c in cat.columns]
            if patt_cols:
                ids_tr = train.meta.get("robot_ids")
                pat_tr = set(map(tuple, cat.set_index("id").loc[ids_tr][patt_cols].astype(str).values.tolist()))
                def _filter_split(split):
                    ids = np.asarray(split.meta.get("robot_ids"))
                    df  = cat.set_index("id")
                    keep = (~df.loc[ids, patt_cols]
                              .astype(str)
                              .apply(tuple, axis=1)
                              .isin(pat_tr)
                             ).to_numpy(dtype=bool)
                    return split.filter(keep)

    train = _dedupe_train_by_robot_ids(train)
    train = _enforce_pattern_limits(train, data, S)
    train = _balance_and_cap_train(train, data, S)
    train = _final_hard_cap_train(train, data, S)

    if S.get("label_noise_rate", 0.0) > 0:
        train = _apply_label_noise(train, S["label_noise_rate"], seed=int(S["seed"]))
        valid = _apply_label_noise(valid, S["label_noise_rate"], seed=int(S["seed"]))
        test  = _apply_label_noise(test,  S["label_noise_rate"], seed=int(S["seed"]))

    print("Train rows:", len(train), "unique_ids:", len(np.unique(train.meta.get('robot_ids'))))
    cat = data.meta.get("catalog_df")
    if cat is not None and S.get("coarse_balance_feature"):
        ids_tr = train.meta.get("robot_ids")
        df_tr = cat.set_index("id").loc[ids_tr]
        coarse = S.get("coarse_balance_feature")
        if coarse in df_tr.columns:
            print("Coarse balance counts:", dict(df_tr[coarse].value_counts().to_dict()))
    if cat is not None:
        ids_tr = train.meta.get("robot_ids")
        df_tr = cat.set_index("id").loc[ids_tr]
        fs_cols = [c for c in df_tr.columns if c.startswith("foot_shape_") and not c.endswith("_subtype")]
        if fs_cols:
            top = df_tr[fs_cols].sum().sort_values(ascending=False).head(10).to_dict()
            print("top_foot_subtypes_train:", top)
        if "foot_shape" in df_tr.columns and "body_shape" in df_tr.columns:
            coarse = df_tr["foot_shape"].astype(str).str.startswith("pointy").astype(int)
            prox = df_tr["body_shape"].map({"round":1,"square":0}).astype(int)
            align = float((prox.values == coarse.values).mean())
            print(f"proxy_align_body_vs_foot: {align:.3f}")
        if "hand_shape" in df_tr.columns and "ears_shape" in df_tr.columns:
            coarse_h = df_tr["hand_shape"].astype(str).str.startswith("edgy").astype(int)
            prox_h = df_tr["ears_shape"].map({"triangle":1,"square":0}).astype(int)
            align_h = float((prox_h.values == coarse_h.values).mean())
            print(f"proxy_align_ears_vs_hand: {align_h:.3f}")

        print("Training set concept distributions:")
        for i, concept_name in enumerate(train.concepts):
            unique, counts = np.unique(train.C[:, i], return_counts=True)
            total = counts.sum()
            dist_str = ", ".join([f"{int(k)}: {v} ({v/total:.1%})" for k, v in dict(zip(unique, counts)).items()])
            print(f"  {concept_name}: {dist_str}")
        print("Test set concept distributions:")
        for i, concept_name in enumerate(test.concepts):
            unique, counts = np.unique(test.C[:, i], return_counts=True)
            total = counts.sum()
            dist_str = ", ".join([f"{int(k)}: {v} ({v/total:.1%})" for k, v in dict(zip(unique, counts)).items()])
            print(f"  {concept_name}: {dist_str}")

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        config = {
            'device': device,
            'batch_size': 8,
            'num_workers': 0,
            'pin_memory': False,
        }

        embed_side = int(getattr(train, "meta", {}).get("resolution", 256))
        cd = ConceptDetector(model=RobotConceptClassifier(num_concepts=train.n_concepts, input_size=embed_side))
        det_name = f"detector_proxy_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"
        if S["load_detector"]:
            mini_train = train.filter(np.array([True] + [False] * (len(train.C) - 1)))
            mini_valid = valid.filter(np.array([True] + [False] * (len(valid.C) - 1)))
            cd.fit(mini_train, mini_valid, freeze=True, embed_params={"device": device}, fit_params={"epochs": 1, "device": "cpu"})
            state = torch.load(S["load_detector"], weights_only=False, map_location="cpu")
            cd.load_state_dict(state)
            det_path = Path(S["load_detector"])
        else:
            _mem("before_cd_fit")
            cd.fit(train, valid,
                embed_params={'shuffle': False, **config},
                fit_params={**config, "epochs": 50, "lr": 1e-3, "patience": 10,
                            "batch_size": config.get("batch_size", 8), "num_workers": 0, "pin_memory": False})
            _mem("after_cd_fit")
            
            det_path = Path(settings["out_dir"]) / (S["run_name"] or "run") / det_name
            torch.save({"model_state_dict": cd.model.state_dict(),
                        "calibration_params": cd.calibration_params}, det_path)

        if int(S.get('fe_harness', 0)) == 1:
            try:
                from scripts.fe_harness import run_fe_harness
                names = list(train.concepts)
                C_tr = train.C.astype(float); C_te = test.C.astype(float)
                y_tr = train.y.astype(int);   y_te = test.y.astype(int)
                try:
                    P_tr = cd.predict_proba(train); P_te = cd.predict_proba(test)
                except Exception:
                    P_tr = cd.predict(train).astype(float); P_te = cd.predict(test).astype(float)
                run_fe_harness(C_tr, y_tr, C_te, y_te, P_tr, P_te, names, table_name="FE 2x2 (proxy setup)")
            except Exception as e:
                print("FE harness unavailable or failed:", e)

        P_tr = cd.predict(train, embed_params={'shuffle': False, **config})
        P_vl = cd.predict(valid, embed_params={'shuffle': False, **config})
        P_te = cd.predict(test,  embed_params={'shuffle': False, **config})
        _mem("after_cd_predicts")
        H_tr = (P_tr > 0.5).astype(np.float32)
        H_te = (P_te > 0.5).astype(np.float32)
        H_vl = (P_vl > 0.5).astype(np.float32)

        fe = FrontEndModel()
        fe_name = f"frontend_proxy_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pkl"
        if S["load_frontend"]:
            with open(S["load_frontend"], "rb") as f:
                fe = pickle.load(f)
            fe_path = Path(S["load_frontend"])
        else:
            Ctr = train.C.astype(np.float32)
            if int(S["impute_missing"]) and np.any(Ctr < 0):
                Cin = Ctr.copy(); m = Cin < 0; Cin[m] = P_tr[m]; fe.fit(Cin, train.y.astype(int))
            else:
                if S.get("CBM_type", "joint") == "sequential":
                    keep = np.all(Ctr >= 0, axis=1)
                    fe.fit(H_tr[keep], train.y[keep].astype(int))
                else:
                    fe.fit(Ctr, train.y.astype(int))
            fe_path = Path(settings["out_dir"]) / (S["run_name"] or "run") / fe_name
            with open(fe_path, "wb") as f:
                pickle.dump(fe, f)

        y_pred_det = fe.predict_proba(H_te)
        y_pred_gt  = fe.predict_proba(test.C.astype(np.float32))
        acc_det = float((y_pred_det.argmax(1) == test.y.astype(int)).mean())
        acc_gt  = float((y_pred_gt.argmax(1)  == test.y.astype(int)).mean())
        concept_acc_mean = float((H_te == test.C).mean())
        
        _mem("before_fe_weights_print")
        print("=== Learned Frontend Weights ===")
        for i, concept in enumerate(test.concepts):
            print(f"  {concept}: {fe.model.coef_[0, i]:.4f}")
        print(f"  bias: {fe.model.intercept_[0]:.4f}")
        _mem("after_fe_weights_print")

        # ... inside main, after FE weights ...
        dnn_stats = {}
        if S.get("train_dnn", 0):
            print("Training baseline DNN...")
            paths_tr = [train.base_dir / p for p in train.X]; ytr = train.y.astype(int)
            paths_te = [test.base_dir / p for p in test.X];   yte = test.y.astype(int)
            dnn_acc, proc, dnn_model = train_eval_image(
                paths_tr, ytr, paths_te, yte,
                model_id=S.get("image_model", "google/vit-tiny-patch16-224"),
                epochs=int(S.get("dnn_epochs", S["epochs"])),
                batch_size=int(S.get("dnn_batch_size", 8)),
                lr=float(S.get("dnn_lr", 5e-5)),
                device=device
            )
            dnn_stats = {"dnn_accuracy": float(dnn_acc)}
            print(f"DNN accuracy: {float(dnn_acc)}")
            dnn_name = f"dnn_proxy_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{bias_tag}.pt"
            dnn_path = Path(settings["out_dir"]) / (S["run_name"] or "run") / dnn_name
            torch.save({"model_state_dict": dnn_model.state_dict(), "processor": proc}, dnn_path)
        # metrics.update(dnn_stats)
        
        _mem("before_interventions")
        print(f"[STEP] start_interventions budgets={S.get('budget',[1,2,3,4,5])}", flush=True)
        intervention_results = {}
        budgets = S.get('budget', [1, 2, 3, 4, 5])
        human_acc = S.get("intervention_accuracy", 1.0)
        for budget in budgets:
            for policy in ["top-1", "top-k"]:
                H_intervened, intervention_stats = apply_interventions(
                    pred_probs=P_te, ground_truth=test.C.astype(int), frontend_model=fe,
                    budget_k=budget, intervention_threshold=S.get("intervention_threshold", 0.5),
                    human_accuracy=human_acc, policy=policy, rng=rng
                )
                y_pred_intervened = fe.predict_proba(H_intervened)
                acc_intervened = float((y_pred_intervened.argmax(1) == test.y.astype(int)).mean())
                key = f"budget_{budget}_{policy}_human_acc_{int(human_acc * 100)}"
                intervention_results[key] = {
                    "accuracy": acc_intervened,
                    "accuracy_gain": acc_intervened - acc_det,
                    "predictions_intervened_on": intervention_stats["samples_intervened_on"],
                    "interventions_rate": intervention_stats["intervention_rate"],
                    "avg_edits_per_intervention": intervention_stats["avg_edits_per_intervention"],
                    "total_concept_checks": intervention_stats["total_concept_checks"],
                    "total_concept_edits_made": intervention_stats["total_concept_edits_made"],
                    "policy": policy, "budget": budget, "human_accuracy": human_acc
                }

        meta_name = f"meta_proxy_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.json"
        metrics_name = f"metrics_proxy_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.json"

        meta = {
            "settings": S,
            "run_dir": str(run_dir),
            "artifacts": {"detector": str(det_path), "frontend": str(fe_path)},
            "splits": {"n_train": int(train.n), "n_valid": int(valid.n), "n_test": int(test.n)},
            "concepts": list(test.concepts),
            "intervention_budgets": budgets,
            "intervention_acc": human_acc,
            "naming_slug": slug,
        }

        metrics = {"cbm_acc_detected": acc_det, "cbm_acc_oracle": acc_gt, "concept_det_acc_mean": concept_acc_mean}
        metrics.update(dnn_stats)
        _mem("before_write_outputs")

        meta_path = run_dir / meta_name
        metrics_path = run_dir / metrics_name
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps({"meta_path": str(meta_path), "metrics_path": str(metrics_path), "detector_path": str(det_path), "frontend_path": str(fe_path)}, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # toggles
    parser.add_argument('--fe-harness', dest='fe_harness', type=int, default=None)
    parser.add_argument('--mode', dest='mode', type=str, choices=['oracle','real'])
    # core config
    parser.add_argument('--draw', dest='draw', type=int)
    parser.add_argument('--image-size', dest='image_size', type=str)
    parser.add_argument('--train-dnn', dest='train_dnn', type=int)
    parser.add_argument('--image-model', dest='image_model', type=str)
    parser.add_argument('--dnn-epochs', dest='dnn_epochs', type=int)
    parser.add_argument('--dnn-batch-size', dest='dnn_batch_size', type=int)
    parser.add_argument('--dnn-lr', dest='dnn_lr', type=float)
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--model-type', dest='model_type', type=str)
    parser.add_argument('--logit-scalar', dest='logit_scalar', type=float)
    parser.add_argument('--logit-intercept', dest='logit_intercept', type=float)
    parser.add_argument('--seed', dest='seed', type=int)
    parser.add_argument('--run-name', dest='run_name', type=str)
    # json-ish args
    parser.add_argument('--drop-concepts', dest='drop_concepts', type=str)
    parser.add_argument('--skew-concept', dest='skew_concept', type=str)
    parser.add_argument('--proxy-spec', dest='proxy_spec', type=str)

    # new args
    parser.add_argument('--samples-per-instance', dest='samples_per_instance', type=int)
    parser.add_argument('--enforce-pattern-limits', dest='enforce_pattern_limits', type=int)
    parser.add_argument('--max-feet-per-pattern', dest='max_feet_per_pattern', type=int)
    parser.add_argument('--max-exposures-per-pattern', dest='max_exposures_per_pattern', type=int)
    parser.add_argument('--disjoint-patterns', dest='disjoint_patterns_across_splits', type=int)
    parser.add_argument('--target-train-unique', dest='target_train_unique', type=int)
    parser.add_argument('--coarse-balance-feature', dest='coarse_balance_feature', type=str)
    parser.add_argument('--proxy-p', dest='proxy_p', type=float)
    parser.add_argument('--subtype-label-bias', dest='subtype_label_bias', type=str)
    parser.add_argument('--grid', dest='grid', type=str)

    args, _ = parser.parse_known_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    for key in ['drop_concepts','skew_concept','proxy_spec','subtype_label_bias','grid']:
        if key in overrides and isinstance(overrides[key], str):
            overrides[key] = json.loads(overrides[key])
    
    if 'grid' in overrides:
        import itertools
        grid = overrides.pop('grid') or {}
        keys = list(grid.keys()); vals = [grid[k] for k in keys]
        base = dict(settings); base.update(overrides)
        for combo in itertools.product(*vals) if keys else [()]:
            cfg = dict(base)
            if keys:
                for k, v in zip(keys, combo):
                    cfg[k] = v
                cfg['run_name'] = f"{base.get('run_name','run')}_{'_'.join(f'{k}={v}' for k,v in zip(keys,combo))}"
            main(cfg)
    else:
        settings.update(overrides)
        main(settings)

