from collections.abc import Sequence
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from concept_benchmark.data import ConceptDataset

from .helper.robot_catalog import (
    ALL_ROBOT_FEATURES,
    OUTCOME_MISSING,
    OUTCOME_NAME,
    generate_robot_catalog,
)
from .helper.utils import model_to_logistic, unlist0
from scipy.special import expit


def create_synthetic_dataset(data_type: str = "image", **kwargs) -> ConceptDataset:
    kind = (data_type or "image").strip().lower()
    if kind != "image":
        raise ValueError("proxy.py supports data_type='image' only")
    return create_robot_image_dataset(**kwargs)


def _coarse_bit(series: pd.Series, source: str, source_to_bit: dict | None) -> pd.Series:
    s = series.astype(str)
    if source_to_bit is not None:
        return s.map(lambda v: int(source_to_bit[v]))
    if source == "foot_shape":
        return s.str.startswith("pointy").astype(int)
    if source == "hand_shape":
        return s.str.startswith("edgy").astype(int)
    raise ValueError(f"Provide source_to_bit for source '{source}'")


def _apply_proxies(catalog_df: pd.DataFrame, proxy_spec: dict | None, rng_seed: int = 0) -> pd.DataFrame:
    if not proxy_spec:
        return catalog_df
    df = catalog_df
    for proxy_name, cfg in proxy_spec.items():
        src = cfg["source"]
        p = float(cfg.get("p", 0.7))
        source_to_bit = cfg.get("source_to_bit", None)
        bit_to_value = cfg["bit_to_value"]
        if src not in df.columns:
            raise ValueError(f"proxy source '{src}' not found")
        src_bit = _coarse_bit(df[src], src, source_to_bit)
        salt = int.from_bytes(hashlib.sha256(proxy_name.encode()).digest()[:4], "big")
        rng = np.random.default_rng(int(rng_seed) + salt)
        use_src = rng.random(len(df)) < p
        rnd = rng.integers(0, 2, size=len(df))
        bit = np.where(use_src, src_bit.to_numpy(dtype=int), rnd).astype(int)
        vals = np.vectorize(bit_to_value.__getitem__)(bit)
        df[proxy_name] = vals
    return df


def create_robot_image_dataset(
    *,
    concepts: dict,
    samples_per_instance: int = 1,
    num_robots: int | None = None,
    size: str = "large",
    resolution: int | None = None,
    output_directory: str | Path = ".static/images",
    draw: bool = False,
    model: str = "",
    model_type: str = "deterministic",
    spurious_features: Sequence[str] | None = None,
    irrelevant_features: Sequence[str] | None = None,
    color_mode: str = "color",
    blur: dict | None = None,
    verbose: bool = False,
    train_concept_detector: bool | None = None,
    epochs: int | None = None,
    **extra_params,
) -> ConceptDataset:
    if not concepts:
        raise ValueError("'concepts' dictionary must be provided and non-empty")
    if not model:
        raise ValueError("'model' expression must be provided")
    spurious_features = list(spurious_features or [])
    irrelevant_features = list(irrelevant_features or [])

    samples_per_instance = int(samples_per_instance)
    if samples_per_instance < 1:
        raise ValueError("'samples_per_instance' must be >= 1")

    if resolution is None:
        if size == "small":
            eff_resolution = 64
        elif size == "medium":
            eff_resolution = 128
        elif size == "large":
            eff_resolution = 256
        else:
            raise ValueError("Invalid size. Use 'small', 'medium', or 'large'.")
    else:
        eff_resolution = int(resolution)

    res = generate_robot_catalog(
        concepts=concepts,
        samples_per_instance=samples_per_instance,
        num_robots=num_robots,
        size=size,
        resolution=eff_resolution,
        output_directory=output_directory,
        draw=draw,
        spurious_features=spurious_features,
        irrelevant_features=irrelevant_features,
        color_mode=color_mode,
        blur=blur,
        return_catalog=True,
        verbose=verbose,
    )
    catalog_df = res[0] if isinstance(res, tuple) else res
    catalog_df = catalog_df.copy()
    catalog_df[OUTCOME_NAME] = OUTCOME_MISSING

    proxy_spec = extra_params.get("proxy_spec", None)
    rng_seed = int(extra_params.get("rng_seed", 0))
    proxy_p_override = extra_params.get("proxy_p", None)
    if proxy_p_override is not None and proxy_spec:
        for k in list(proxy_spec.keys()):
            try:
                proxy_spec[k]["p"] = float(proxy_p_override)
            except Exception:
                pass
    catalog_df = _apply_proxies(catalog_df, proxy_spec, rng_seed=rng_seed)

    if "foot_shape" in catalog_df.columns and "foot_shape_subtype" in catalog_df.columns:
        fs = catalog_df["foot_shape"].astype(str)
        fss = catalog_df["foot_shape_subtype"].astype(str)
        for coarse in ["flat", "pointy"]:
            maskc = fs.eq(coarse)
            for subtype in fss.unique():
                col = f"foot_shape_{coarse}_{subtype}"
                catalog_df[col] = (maskc & fss.eq(subtype)).astype(np.int8)
        ids = catalog_df["id"].astype(str)
        hb = ids.map(lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big") & 1).astype(int)
        for coarse in ["flat", "pointy"]:
            maskc = fs.eq(coarse)
            for subtype in fss.unique():
                base = maskc & fss.eq(subtype)
                catalog_df[f"foot_shape_{coarse}_{subtype}_vertex"] = (base & hb.eq(1)).astype(np.int8)
                catalog_df[f"foot_shape_{coarse}_{subtype}_side"] = (base & hb.eq(0)).astype(np.int8)
        catalog_df["foot_orientation"] = np.where(hb.values == 1, "vertex", "side")

    if "hand_shape" in catalog_df.columns and "hand_shape_subtype" in catalog_df.columns:
        hs = catalog_df["hand_shape"].astype(str)
        hss = catalog_df["hand_shape_subtype"].astype(str)
        for coarse in ["round", "edgy"]:
            maskc = hs.eq(coarse)
            for subtype in hss.unique():
                col = f"hand_shape_{coarse}_{subtype}"
                catalog_df[col] = (maskc & hss.eq(subtype)).astype(np.int8)

    std_feats = [f for f in catalog_df.columns if f in ALL_ROBOT_FEATURES]
    sub_feats = [f for f in catalog_df.columns
                 if (f.startswith("foot_shape_") or f.startswith("hand_shape_"))
                 and not f.endswith("_subtype")]
    sub_feats = [f for f in sub_feats if (catalog_df[f] == 1).any()]
    feature_names = std_feats + sub_feats

    pos_map = {}
    for feat in std_feats:
        base = ALL_ROBOT_FEATURES[feat][0]
        pos_map[feat] = base.split("_")[0] if isinstance(base, str) else base
    for feat in sub_feats:
        pos_map[feat] = 1

    cols = list(pos_map.keys())
    vals = list(pos_map.values())
    C = (catalog_df[cols] == vals).to_numpy().astype(np.int8)

    if model_type == "deterministic":
        label_str = lambda row: eval(unlist0(model))
        catalog_df[OUTCOME_NAME] = catalog_df.apply(label_str, axis=1).map({"glorp": 1, "drent": 0}).astype(int)
        y = catalog_df[OUTCOME_NAME].to_numpy()
    elif model_type == "stochastic":
        rng = np.random.default_rng(int(extra_params.get("rng_seed", 0)))
        prob_fun = lambda row: float(eval(
            model_to_logistic(
                model,
                scalar=float(extra_params.get("scalar", 1.0)),
                intercept=extra_params.get("intercept", None)
            )
        ))
        catalog_df["_p_base"] = catalog_df.apply(prob_fun, axis=1).clip(0, 1)
        bias_cfg = extra_params.get("subtype_label_bias", {})
        delta = np.zeros(len(catalog_df), dtype=float)
        if isinstance(bias_cfg, dict) and len(bias_cfg) > 0:
            bad = [k for k in bias_cfg.keys() if k not in catalog_df.columns]
            if bad and verbose:
                print("unknown subtype_label_bias keys:", bad)
            for col_name, logit_delta in bias_cfg.items():
                if col_name in catalog_df.columns:
                    v = float(logit_delta)
                    delta += catalog_df[col_name].astype(int).to_numpy() * v
        logits = np.log(catalog_df["_p_base"] / (1.0 - catalog_df["_p_base"] + 1e-12) + 1e-12) + delta
        p_adj = expit(logits)
        y = rng.binomial(1, p_adj.clip(0, 1))
        catalog_df[OUTCOME_NAME] = y
        if verbose:
            print(f"class_rate_glorp: {float(y.mean()):.3f}")
            if "foot_shape" in catalog_df.columns:
                cc = catalog_df["foot_shape"].astype(str).str.startswith("pointy").value_counts().to_dict()
                print("coarse_pointy_counts:", cc)
    else:
        raise ValueError("Invalid model_type. Use 'deterministic' or 'stochastic'.")

    image_dir = output_directory
    X = np.array([row["png_filename"] for _, row in catalog_df.iterrows()])

    total_robots = int(len(catalog_df))
    meta = {
        "classes": ["drent", "glorp"],
        "concepts": feature_names,
        "data_type": "image",
        "image_dir": image_dir,
        "resolution": eff_resolution,
        "color_mode": color_mode,
        "labeling_function": model,
        "num_robots": total_robots,
        "num_unique_robots": int(pd.Series(catalog_df["id"]).nunique()),
        "robot_ids": catalog_df["id"].values,
        "catalog_df": catalog_df,
    }

    return ConceptDataset(X=X, C=C, y=y, meta=meta, base_dir=image_dir)
