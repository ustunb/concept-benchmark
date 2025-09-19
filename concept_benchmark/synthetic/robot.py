import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from concept_benchmark.data import ConceptDataset

from .helper.robot_catalog import (
    ALL_ROBOT_FEATURES,
    OUTCOME_MISSING,
    OUTCOME_NAME,
    generate_robot_catalog,
)
from .helper import textgen as text_helper
from .helper.utils import model_to_logistic, unlist0


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _default_eval_globals() -> dict[str, Any]:
    return {
        "np": np,
        "numpy": np,
        "expit": _sigmoid,
        "sigmoid": _sigmoid,
        "int": int,
        "float": float,
        "bool": bool,
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
    }


def _value_to_probability(
    value: Any,
    *,
    positive_label: str = "glorp",
    negative_label: str = "drent",
) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if positive_label and lowered == positive_label.lower():
            return 1.0
        if negative_label and lowered == negative_label.lower():
            return 0.0
        try:
            value = float(value)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                "Could not interpret label model output as probability"
            ) from exc
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(
        "Label model must return a string label, boolean, or probability value"
    )


def _clip_probability(p: float) -> float:
    return float(np.clip(p, 0.0, 1.0))


def _bayes_accuracy(probabilities: np.ndarray) -> float:
    return float(np.mean(np.maximum(probabilities, 1.0 - probabilities)))


def _apply_flip_noise(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    if not 0.0 <= epsilon <= 0.5:
        raise ValueError("flip probability epsilon must be between 0 and 0.5")
    if epsilon == 0.0:
        return probabilities
    return probabilities * (1.0 - 2.0 * epsilon) + epsilon


def _solve_flip_probability(
    base_probabilities: np.ndarray,
    target_accuracy: float,
) -> float:
    base_accuracy = _bayes_accuracy(base_probabilities)
    if base_accuracy <= 0.5 + 1e-8:
        raise ValueError("Base label model must yield Bayes accuracy above random chance")
    if not 0.5 <= target_accuracy <= base_accuracy + 1e-8:
        raise ValueError(
            f"target_accuracy must be in [0.5, {base_accuracy:.3f}] for symmetric flip noise"
        )
    if target_accuracy >= base_accuracy:
        return 0.0
    numerator = target_accuracy - 0.5
    denominator = base_accuracy - 0.5
    ratio = numerator / denominator
    epsilon = 0.5 * (1.0 - ratio)
    return float(np.clip(epsilon, 0.0, 0.5))


def estimate_bayes_accuracy(probabilities: Sequence[float]) -> float:
    """Compute the Bayes optimal accuracy implied by label probabilities."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("Probabilities must be a 1D iterable")
    return _bayes_accuracy(probs)


def symmetric_flip_probability_for_target(
    probabilities: Sequence[float], target_accuracy: float
) -> float:
    """Solve the symmetric flip noise needed to reach ``target_accuracy``."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("Probabilities must be a 1D iterable")
    return _solve_flip_probability(probs, target_accuracy)


@runtime_checkable
class RobotLabelModel(Protocol):
    def proba(self, row: pd.Series) -> float:
        """Return P(label=1 | row)."""


@dataclass
class ExpressionLabelModel(RobotLabelModel):
    expression: str
    mode: str = "deterministic"
    positive_label: str = "glorp"
    negative_label: str = "drent"
    global_namespace: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        expr = unlist0(self.expression)
        self._compiled = compile(expr, "<robot_label_model>", "eval")
        base_globals = _default_eval_globals()
        if self.global_namespace:
            base_globals.update(self.global_namespace)
        self._globals = {"__builtins__": {}} | base_globals

    def proba(self, row: pd.Series) -> float:
        result = eval(self._compiled, self._globals, {"row": row})
        if self.mode == "stochastic":
            prob = _value_to_probability(result)
        else:
            prob = _value_to_probability(
                result, positive_label=self.positive_label, negative_label=self.negative_label
            )
            prob = 1.0 if prob >= 0.5 else 0.0
        return _clip_probability(prob)


@dataclass
class CallableLabelModel(RobotLabelModel):
    fn: Callable[[pd.Series], Any]
    positive_label: str = "glorp"
    negative_label: str = "drent"

    def proba(self, row: pd.Series) -> float:
        value = self.fn(row)
        prob = _value_to_probability(
            value, positive_label=self.positive_label, negative_label=self.negative_label
        )
        return _clip_probability(prob)


@dataclass
class NoisyLabelModel(RobotLabelModel):
    base: RobotLabelModel
    flip_probability: float | Callable[[pd.Series], float]

    def proba(self, row: pd.Series) -> float:
        p = _clip_probability(self.base.proba(row))
        epsilon = self.flip_probability(row) if callable(self.flip_probability) else self.flip_probability
        epsilon = float(np.clip(epsilon, 0.0, 0.5))
        return _clip_probability(p * (1.0 - 2.0 * epsilon) + epsilon)


def _coerce_label_model(
    model: Any,
    *,
    model_type: str,
) -> RobotLabelModel:
    if isinstance(model, (list, tuple)):
        if not model:
            raise ValueError("Empty model specification provided")
        return _coerce_label_model(model[0], model_type=model_type)
    if isinstance(model, RobotLabelModel):
        return model
    if callable(model):
        return CallableLabelModel(model)
    if isinstance(model, str):
        expr = unlist0(model)
        if model_type == "stochastic" and ">=" in expr and "expit" not in expr:
            expr = model_to_logistic(expr)
        return ExpressionLabelModel(expr, mode=model_type)
    raise TypeError("model must be a string, callable, or RobotLabelModel instance")


def create_synthetic_dataset(data_type: str = "image", **kwargs) -> ConceptDataset:
    """Factory that creates either an image or text robot dataset.

    Args:
        data_type: ``"image"`` to call :func:`create_synthetic_dataset`,
            ``"text"`` to call :func:`create_robot_text_dataset`.
        **kwargs: Forwarded to the respective underlying builder.

    Returns:
        ConceptDataset generated by the chosen modality.
    """

    kind = (data_type or "image").strip().lower()
    if kind == "image":
        return create_robot_image_dataset(**kwargs)
    if kind == "text":
        return create_robot_text_dataset(**kwargs)
    raise ValueError("data_type must be either 'image' or 'text'")


def create_robot_text_dataset(
    source,
    templates: Sequence[str] | None = None,
    variants_per_row: int = 3,
    include_color: bool = True,
    rng_seed: int = 0,
    concept_noise: float | None = None,
    head_col: str = "head_shape",
    body_col: str = "body_shape",
    knees_col: str = "has_knees",
    elbows_col: str = "has_elbows",
    foot_col: str = "foot_shape",
    color_mode_col: str = "color_mode",
    concept_cols: Iterable[str] | None = None,
    label_col: str | None = None,
    label_map: dict | None = None,
    drop_unknown: bool = True,
    text_mode: str | None = None,
    use_llm: bool = False,
    llm_provider: str = "gemini",
    llm_model: str = "gemini-1.5-flash",
    llm_api_key: str | None = None,
    llm_system: str | None = None,
    llm_user_prompt: str | None = None,
) -> ConceptDataset:
    if templates is None or text_mode == "structured":
        templates = text_helper.DEFAULT_TEMPLATES
    templates = [re.sub(r'^[\uFEFF\u200B-\u200D]+', '', t) for t in templates]
    templates = [text_helper._rewrite_modifiers(t) for t in templates]
    rng = np.random.default_rng(rng_seed)
    if concept_noise is not None:
        if not 0.0 <= concept_noise <= 1.0:
            raise ValueError("concept_noise must be within [0.0, 1.0]")
        concept_noise_p = float(concept_noise)
    else:
        concept_noise_p = 0.0
    mode = (text_mode or ("llm" if use_llm else "unstructured")).strip().lower()

    if isinstance(source, ConceptDataset):
        df = getattr(source, "catalog_df", None)
        if df is None:
            raise ValueError("ConceptDataset missing catalog_df")
        C = np.asarray(source.C)
        y = np.asarray(source.y)
        concept_names = list(
            source.meta.get("concepts", [f"c{i}" for i in range(C.shape[1])])
        )
        classes = list(
            source.meta.get("classes", sorted(map(int, np.unique(y))))
        )
    else:
        if not isinstance(source, pd.DataFrame):
            raise TypeError("source must be ConceptDataset or pd.DataFrame")
        if concept_cols is None or label_col is None:
            raise ValueError(
                "Provide concept_cols and label_col when source is a DataFrame"
            )
        df = source
        C, concept_names = text_helper._binarize_concepts(df, concept_cols)
        y = text_helper._to_label(df[label_col].to_numpy(), label_map)
        if label_map is not None:
            classes = [str(k) for k, v in sorted(label_map.items(), key=lambda kv: kv[1])]
        else:
            raw_uniqs = (
                pd.Series(df[label_col]).astype(str).str.lower().unique().tolist()
            )
            classes = (
                ["drent", "glorp"]
                if {"glorp", "drent"} <= set(raw_uniqs)
                else [str(v) for v in sorted(np.unique(y).tolist())]
            )

    X, idxs = [], []
    tbool = (
        lambda v: (v.lower() in {"true", "t", "yes", "y", "1"})
        if isinstance(v, str)
        else bool(v)
    )
    colorish = (
        lambda d: (
            "color"
            if any(
                c in d.columns
                for c in (
                    "color",
                    "left_color",
                    "right_color",
                    "primary_color",
                    "secondary_color",
                )
            )
            else "greyscale"
        )
    )

    def _colors_for_row(r):
        if "left_color" in df.columns and "right_color" in df.columns:
            return str(r["left_color"]), str(r["right_color"])
        if "primary_color" in df.columns and "secondary_color" in df.columns:
            return str(r["primary_color"]), str(r["secondary_color"])
        if "color1" in df.columns and "color2" in df.columns:
            return str(r["color1"]), str(r["color2"])
        return None, None

    structured_templates_default = [
        "This robot has a {head_shape} head and a {body_shape} body. It {has_elbows} and {has_knees}. Its feet are {foot_shape}.",
        "Head: {head_shape}. Body: {body_shape}. Elbows: {has_elbows}. Knees: {has_knees}. Feet: {foot_shape}.",
    ]

    for i, row in df.iterrows():
        cms = (
            row.get(color_mode_col, None)
            if (include_color and color_mode_col in df.columns)
            else (colorish(df) if include_color else "greyscale")
        )
        knees_b = tbool(row.get(knees_col, False))
        elbows_b = tbool(row.get(elbows_col, False))
        ant_b = tbool(row.get("has_antennae", False))
        c1, c2 = _colors_for_row(row)
        if c1 and c2:
            color_pair = f"{c1} and {c2}"
            color_single = f"{c1}/{c2}"
        else:
            color_pair = str(row.get("color")) if "color" in df.columns else None
            color_single = color_pair
        fill = {
            "head_shape": str(row.get(head_col, "")).replace("_", " "),
            "body_shape": str(row.get(body_col, "")).replace("_", " "),
            "foot_shape": str(row.get(foot_col, "")).replace("_", " "),
            "ears_shape": str(row.get("ears_shape", "")).replace("_", " "),
            "mouth_type": str(row.get("mouth_type", "")).replace("_", " "),
            "hand_shape": str(row.get("hand_shape", "")).replace("_", " "),
            "has_knees": "has knees" if knees_b else "no knees",
            "has_elbows": "has elbows" if elbows_b else "no elbows",
            "has_antennae": "has antennae" if ant_b else "no antennae",
            "has_knees_word": "has" if knees_b else "no",
            "has_elbows_word": "has" if elbows_b else "no",
            "has_antennae_word": "has" if ant_b else "no",
            "has_knees_bool": "true" if knees_b else "false",
            "has_elbows_bool": "true" if elbows_b else "false",
            "has_antennae_bool": "true" if ant_b else "false",
            "color": color_single if color_single else "unknown",
            "color_pair": color_pair if color_pair else "unknown",
            "color_left": c1 if c1 else "unknown",
            "color_right": c2 if c2 else "unknown",
            "color_mode": "greyscale" if not include_color else str(cms),
        }
        fill["hand_shape"] = text_helper._synonym("hand_shape", fill["hand_shape"])
        fill["foot_shape"] = text_helper._synonym("foot_shape", fill["foot_shape"])
        for name in (
            "head_shape",
            "body_shape",
            "ears_shape",
            "mouth_type",
            "hand_shape",
            "foot_shape",
        ):
            val = fill.get(name, "")
            fill[name + "_syn"] = text_helper._synonym(name, val)
            neg = text_helper._negate(name, val)
            fill[name + "_not"] = (
                str(neg).replace("_", " ") if neg is not None else f"not {val}"
            )
        fill["has_knees_not"] = "no knees" if knees_b else "with knees"
        fill["has_elbows_not"] = "no elbows" if elbows_b else "with elbows"
        fill["has_antennae_not"] = "no antennae" if ant_b else "with antennae"
        sfill = text_helper._Safe(fill)

        if mode == "llm":
            if concept_cols is None:
                concept_keys = [
                    head_col,
                    body_col,
                    foot_col,
                    knees_col,
                    elbows_col,
                    "has_antennae",
                    "mouth_type",
                    "ears_shape",
                    "hand_shape",
                    "color",
                ]
                concept_keys = [k for k in concept_keys if k in df.columns]
            else:
                concept_keys = list(concept_cols)
            concepts_dict = {k: (str(row[k]) if k in row else "") for k in concept_keys}
            llm_texts = text_helper.unstructured_caption_via_llm(
                concepts=concepts_dict,
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                system=llm_system,
                user_prompt=llm_user_prompt,
                n=variants_per_row,
            )
            for out in llm_texts:
                out = text_helper._polish_text(out)
                X.append(out)
                idxs.append(i)
        elif mode == "structured":
            base_tpls = templates if templates else structured_templates_default
            for tpl in rng.choice(base_tpls, size=variants_per_row, replace=True):
                s = text_helper._rewrite_modifiers(tpl)
                out = s.format_map(sfill)
                if drop_unknown:
                    out = text_helper._clean_unknown(out)
                out = text_helper._polish_text(out)
                X.append(out)
                idxs.append(i)
        else:
            for tpl in rng.choice(templates, size=variants_per_row, replace=True):
                s = text_helper._rewrite_modifiers(tpl)
                out = s.format_map(sfill)
                if drop_unknown:
                    out = text_helper._clean_unknown(out)
                out = text_helper._polish_text(out)
                X.append(out)
                idxs.append(i)

    idxs = np.asarray(idxs, dtype=int)
    C_out = C[idxs]
    y_out = y[idxs]
    meta = {
        "data_type": "text",
        "templates": list(templates),
        "concepts": concept_names,
        "classes": classes,
        "row_index": idxs,
    }
    if concept_noise is not None:
        meta["concept_noise"] = {
            "enabled": concept_noise_p > 0.0,
            "scheme": "uniform_flip",
            "p": concept_noise_p,
        }

    dataset = ConceptDataset(
        X=list(X),
        C=np.asarray(C_out, dtype=np.int8),
        y=np.asarray(y_out, dtype=np.int32),
        meta=meta,
    )

    if concept_noise_p > 0.0:
        dataset.sample_concept_noise(p=concept_noise_p, rng=rng_seed, enable=True)

    return dataset


def create_robot_image_dataset(
    *,
    concepts: dict,
    n: int = 1,
    num_robots: int | None = None,
    size: str = "large",
    resolution: int | None = None,
    output_directory: str | Path = ".static/images",
    draw: bool = False,
    model: str = "",
    model_type: str = "deterministic",
    target_accuracy: float | None = None,
    concept_noise: float | None = None,
    rng_seed: int | None = 0,
    spurious_features: Sequence[str] | None = None,
    irrelevant_features: Sequence[str] | None = None,
    color_mode: str = "color",
    blur: dict | None = None,
    verbose: bool = False,
    train_concept_detector: bool | None = None,
    epochs: int | None = None,
    **extra_params,
) -> ConceptDataset:
    """Create an image-based robot ConceptDataset.

    Args:
        concepts: Mapping from concept name to allowed values.
        n: Number of sampled robots per concept configuration (previously `samples_per_instance`).
        num_robots: Explicit number of robots to generate.
        size: Rendering size key.
        resolution: Override resolution.
        output_directory: Directory where rendered images are stored.
        draw: Whether to render robots to disk.
        model: String expression, callable, or :class:`RobotLabelModel` that returns
            label probabilities. Strings are evaluated with ``row`` bound to the
            catalog row; deterministic expressions should emit ``'glorp'``/``'drent'``
            (or booleans), stochastic expressions should emit probabilities.
        model_type: "deterministic" skips label sampling and expects 0/1 outcomes;
            "stochastic" samples labels from the provided probabilities.
        target_accuracy: Optional desired Bayes optimal accuracy. If provided, the
            routine applies symmetric label-flip noise to match the target (down to
            chance level 0.5).
        concept_noise: Probability of flipping each concept bit independently.
            Must be in [0.0, 1.0].
        rng_seed: Seed used when sampling stochastic labels.
        spurious_features: Features treated as spurious.
        irrelevant_features: Additional features to drop from the catalog.
        color_mode: Rendering color mode.
        verbose: Print catalog debug information.

    Returns:
        A :class:`ConceptDataset` with metadata describing the labeling process.
    """

    if not concepts:
        raise ValueError("'concepts' dictionary must be provided and non-empty")
    if not model:
        raise ValueError("'model' expression must be provided for label generation")

    num_combinations = int(np.prod([len(v) for v in concepts.values()]))
    total_robots = num_robots or num_combinations * n
    eff_resolution = resolution if resolution is not None else (600 if size == "large" else 36)
    spurious = list(spurious_features or [])
    irrelevant = list(irrelevant_features) if irrelevant_features is not None else spurious
    drop_irrelevant = extra_params.pop("drop_irrelevant", True)
    _ = (train_concept_detector, epochs)  # parameters accepted for API compatibility

    catalog_df = generate_robot_catalog(
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
    catalog_df = catalog_df.copy()
    catalog_df[OUTCOME_NAME] = OUTCOME_MISSING
    df = catalog_df

    if model_type not in {"deterministic", "stochastic"}:
        raise ValueError("Invalid model_type. Use 'deterministic' or 'stochastic'.")

    label_model = _coerce_label_model(model, model_type=model_type)

    base_probabilities = []
    for _, row in df.iterrows():
        base_probabilities.append(_clip_probability(label_model.proba(row)))
    base_probabilities = np.asarray(base_probabilities, dtype=np.float64)

    epsilon = 0.0
    probabilities = base_probabilities
    if target_accuracy is not None:
        epsilon = _solve_flip_probability(base_probabilities, target_accuracy)
        probabilities = _apply_flip_noise(base_probabilities, epsilon)

    if concept_noise is not None:
        if not 0.0 <= concept_noise <= 1.0:
            raise ValueError("concept_noise must be within [0.0, 1.0]")
        concept_noise_p = float(concept_noise)
    else:
        concept_noise_p = 0.0

    rng = np.random.default_rng(rng_seed)
    sample_labels = model_type == "stochastic" or (target_accuracy is not None and epsilon > 0.0)
    if sample_labels:
        labels = rng.binomial(1, np.clip(probabilities, 0.0, 1.0)).astype(np.int32)
    else:
        labels = (probabilities >= 0.5).astype(np.int32)

    catalog_df["glorp_probability_base"] = base_probabilities.astype(np.float32)
    catalog_df["glorp_probability"] = probabilities.astype(np.float32)
    catalog_df[OUTCOME_NAME] = labels

    if verbose:
        print("Catalog DataFrame:")
        print(catalog_df.to_string(index=False))

    # X: Image paths (stored as strings)
    image_dir = output_directory
    X = np.array([row["png_filename"] for _, row in catalog_df.iterrows()])

    # C: Concept matrix
    feature_names = [
        feat for feat in catalog_df.columns if feat in ALL_ROBOT_FEATURES
    ]
    pos_map = {
        feat: ALL_ROBOT_FEATURES[feat][0].split("_")[0] \
        if isinstance(ALL_ROBOT_FEATURES[feat][0], str) \
        else ALL_ROBOT_FEATURES[feat][0]
        for feat in feature_names
    }
    C = (catalog_df[pos_map.keys()] == pos_map.values()).to_numpy().astype(np.int8)

    # y: Labels pr P(y=1|x)
    y = catalog_df[OUTCOME_NAME].values

    if verbose:
        print("Dataset for Training:")
        print(X)
        print(C)
        print(y)

    # colors to string (colors don't play well with pickle)
    catalog_df['color_left'] = catalog_df['color_left'].astype(str)
    catalog_df['color_right'] = catalog_df['color_right'].astype(str)

    base_accuracy = _bayes_accuracy(base_probabilities)
    bayes_accuracy = _bayes_accuracy(probabilities)
    label_descriptor = model if isinstance(model, str) else repr(model)
    label_stats = {
        "model_type": model_type,
        "base_bayes_accuracy": base_accuracy,
        "bayes_accuracy": bayes_accuracy,
    }
    if target_accuracy is not None:
        label_stats["target_accuracy"] = float(target_accuracy)
    if epsilon:
        label_stats["flip_probability"] = float(epsilon)
    if rng_seed is not None:
        label_stats["rng_seed"] = int(rng_seed)

    # Meta: metadata for ConceptDataset
    meta = {
        "classes": ["drent", "glorp"],
        "concepts": feature_names,
        "data_type": "image",
        "image_dir": image_dir,
        "resolution": eff_resolution,
        "color_mode": color_mode,
        "label_model": label_descriptor,
        "labeling_function": label_descriptor,
        "label_stats": label_stats,
        "num_robots": total_robots,
        "robot_ids": catalog_df["id"].values,
        "catalog_df": catalog_df,
    }

    if concept_noise is not None:
        meta["concept_noise"] = {
            "enabled": concept_noise_p > 0.0,
            "scheme": "uniform_flip",
            "p": concept_noise_p,
        }

    robot_dataset = ConceptDataset(
        X=X,
        C=C,
        y=y,
        meta=meta,
        base_dir=image_dir,
    )

    if concept_noise_p > 0.0:
        robot_dataset.sample_concept_noise(p=concept_noise_p, rng=rng_seed, enable=True)

    return robot_dataset

# Sample kwargs:

# if __name__ == "__main__":
#     params = {
#         'samples_per_instance': 1,
#         # how many times to repeat each robot with changed colors (irrelavant feature); max 108
#         'draw': True,
#         'output_directory': './robot_images',
#         'concepts': {
#             'head_shape': ['square', 'round'],
#             'body_shape': ['square', 'round'],
#             'has_knees': ['false', 'true'],
#             'has_elbows': ['false', 'true'],
#             'has_antennae': ['false', 'true'],
#             'ears_shape': ['square', 'triangle'],
#             'mouth_type': ['closed', 'open'],
#             'hand_shape': ['round_circle', 'round_oval', 'round_oval2',
#                            'edgy_triangle', 'edgy_square', 'edgy_trapezoid'],
#             'foot_shape': ['flat_4sided', 'flat_5sided', 'flat_lshaped',
#                            'pointy_3sided', 'pointy_4sided', 'pointy_6sided'],
#         },
#         'spurious_features': ['has_elbows', 'hand_shape'],  # features that do not appear in the catalog + color
#         'model': "'glorp' if (int(row['body_shape']=='square') + int(row['foot_shape']=='pointy') - 2 >= 0) else 'drent'",
#         'model_type': 'deterministic',  # 'deterministic', 'stochastic'
#         'size': 'large',  # 'small', 'large'
#         'color_mode': 'color',  # 'greyscale', 'color'
#     }
#
#     dataset = create_synthetic_dataset(**params)
#     print(dataset)
