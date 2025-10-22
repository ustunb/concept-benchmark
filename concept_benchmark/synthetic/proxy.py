from collections.abc import Sequence
from pathlib import Path

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


def create_synthetic_dataset(data_type: str = "image", **kwargs) -> ConceptDataset:
    kind = (data_type or "image").strip().lower()
    if kind != "image":
        raise ValueError("proxy.py supports data_type='image' only")
    return create_robot_image_dataset(**kwargs)


def _coarse_bit(series: pd.Series, source: str, source_to_bit: dict | None) -> pd.Series:
    s = series.astype(str)
    if source_to_bit is not None:
        out = s.map(source_to_bit)  # try exact values (coarse or subtype)
        if out.isna().any():
            # try mapping the coarse token (prefix before "_")
            coarse = s.str.split("_").str[0]
            out2 = coarse.map(source_to_bit)
            if out2.isna().any():
                # final fallback: infer by prefix
                if source == "foot_shape":
                    return s.str.startswith("pointy").astype(int)
                if source == "hand_shape":
                    return s.str.startswith("edgy").astype(int)
                raise ValueError(f"source_to_bit missing mapping for '{source}' values: {series.unique().tolist()}")
            return out2.astype(int)
        return out.astype(int)
    # no mapping provided → infer by prefix
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
        rng = np.random.default_rng(int(rng_seed) + (hash(proxy_name) & 0xFFFFFFFF))
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
        raise ValueError("'model' expression must be provided for label generation")

    num_combinations = int(np.prod([len(v) for v in concepts.values()]))
    total_robots = num_robots or num_combinations * samples_per_instance
    eff_resolution = resolution if resolution is not None else (600 if size == "large" else 36)
    spurious = list(spurious_features or [])
    irrelevant = list(irrelevant_features) if irrelevant_features is not None else spurious
    drop_irrelevant = extra_params.pop("drop_irrelevant", True)
    _ = (train_concept_detector, epochs)

    res = generate_robot_catalog(
        concepts=concepts,
        num_robots=total_robots,
        resolution=eff_resolution,
        output_directory=output_directory,
        draw=draw,
        color_mode=color_mode,
        blur=blur,
        drop_irrelevant=drop_irrelevant,
        irrelevant_features=irrelevant,
        verbose=verbose,
        **extra_params,
    )
    catalog_df = res[0] if isinstance(res, tuple) else res
    catalog_df = catalog_df.copy()
    catalog_df[OUTCOME_NAME] = OUTCOME_MISSING

    # Proxies: correlate to coarse sources with prob p (post-catalog; images will not reflect proxy flips if draw=True)
    proxy_spec = extra_params.get("proxy_spec", None)
    rng_seed = int(extra_params.get("rng_seed", 0))
    catalog_df = _apply_proxies(catalog_df, proxy_spec, rng_seed=rng_seed)

    df = catalog_df

    if model_type == "deterministic":
        glorp_model_true = lambda row: eval(unlist0(model))
    elif model_type == "stochastic":
        glorp_model_true = lambda row: eval(model_to_logistic(model))
    else:
        raise ValueError("Invalid model_type. Use 'deterministic' or 'stochastic'.")

    df[OUTCOME_NAME] = df.apply(glorp_model_true, axis=1)
    catalog_df[OUTCOME_NAME] = catalog_df.apply(glorp_model_true, axis=1)

    if model_type == "deterministic":
        catalog_df[OUTCOME_NAME] = catalog_df[OUTCOME_NAME].apply(lambda x: 1 if x == "glorp" else 0)

    if verbose:
        print("Catalog DataFrame:")
        print(catalog_df.to_string(index=False))

    image_dir = output_directory
    X = np.array([row["png_filename"] for _, row in catalog_df.iterrows()])

    feature_names = [feat for feat in catalog_df.columns if feat in ALL_ROBOT_FEATURES]
    pos_map = {
        feat: ALL_ROBOT_FEATURES[feat][0].split("_")[0]
        if isinstance(ALL_ROBOT_FEATURES[feat][0], str)
        else ALL_ROBOT_FEATURES[feat][0]
        for feat in feature_names
    }
    C = (catalog_df[pos_map.keys()] == pos_map.values()).to_numpy().astype(np.int8)

    y = catalog_df[OUTCOME_NAME].values

    if verbose:
        print("Dataset for Training:")
        print(X)
        print(C)
        print(y)

    catalog_df["color_left"] = catalog_df["color_left"].astype(str)
    catalog_df["color_right"] = catalog_df["color_right"].astype(str)

    meta = {
        "classes": ["drent", "glorp"],
        "concepts": feature_names,
        "data_type": "image",
        "image_dir": image_dir,
        "resolution": eff_resolution,
        "color_mode": color_mode,
        "labeling_function": model,
        "num_robots": total_robots,
        "robot_ids": catalog_df["id"].values,
        "catalog_df": catalog_df,
    }

    return ConceptDataset(X=X, C=C, y=y, meta=meta, base_dir=image_dir)
