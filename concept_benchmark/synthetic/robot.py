import numpy as np

from concept_benchmark.data import ConceptDataset
from concept_benchmark.paths import results_dir

from .helper.robot_catalog import RobotDistribution, generate_robot_catalog
from .helper.utils import model_to_logistic, unlist0


def create_synthetic_dataset(**kwargs):
    """
    Create synthetic robot dataset that returns ConceptDataset

    Args:
        **kwargs: Parameters for robot generation (same as your existing params)

    Returns:
        RobotConceptDataset object
    """
    num_combinations = int(np.prod([len(v) for v in kwargs["concepts"].values()]))
    kwargs["num_robots"] = kwargs.get("num_robots", num_combinations) * \
        kwargs.get("samples_per_instance", 1)
    kwargs["resolution"] = 600 if kwargs.get("size", "large") == "large" else 32
    kwargs["irrelevant_features"] = kwargs.get("spurious_features", [])

    catalog_df = generate_robot_catalog(kwargs)
    rdist = RobotDistribution(df=catalog_df)
    df = rdist.df

    # Specify true labels
    if kwargs.get("model_type", "deterministic") == "deterministic":
        glorp_model_true = lambda row: eval(unlist0(kwargs["model"]))
    elif kwargs.get("model_type", "deterministic") == "stochastic":
        glorp_model_true = lambda row: eval(model_to_logistic(kwargs["model"]))
    else:
        raise ValueError("Invalid model_type. Use 'deterministic' or 'stochastic'.")

    df[rdist.outcome_name] = df.apply(glorp_model_true, axis=1)
    catalog_df[rdist.outcome_name] = catalog_df.apply(glorp_model_true, axis=1)

    if kwargs.get("model_type", "deterministic") == "deterministic":
        # change "glorp" to 1 and "drent" to 0
        catalog_df[rdist.outcome_name] = catalog_df[rdist.outcome_name].apply(
            lambda x: 1 if x == "glorp" else 0
        )

    if kwargs.get("verbose", "False"):
        print("Catalog DataFrame:")
        print(catalog_df.to_string(index=False))

    # X: Image paths (stored as strings)
    image_dir = kwargs.get("output_directory", ".static/images")
    X = np.array([row["png_filename"] for _, row in catalog_df.iterrows()])

    # C: Concept matrix
    feature_names = rdist.feature_names
    pos_map = rdist.positive_value_by_feature
    # Binary encode concepts: 1 if feature equals designated positive value, else 0
    C_cols = []
    for feat in feature_names:
        pos_val = pos_map.get(feat)
        col = (
            (catalog_df[feat].astype(str).str.split("_").str[0] == str(pos_val))
            .astype(np.int32)
            .to_numpy()
        )
        C_cols.append(col)
    C = np.stack(C_cols, axis=1).astype(np.int8)

    # y: Labels pr P(y=1|x)
    y = catalog_df[rdist.outcome_name].values

    if kwargs.get("verbose", "False"):
        print("Dataset for Training:")
        print(X)
        print(C)
        print(y)

    # Meta: metadata for ConceptDataset
    meta = {
        "classes": ["drent", "glorp"],
        "concepts": feature_names,
        "data_type": "image",
        "image_dir": image_dir,
        "resolution": kwargs.get("resolution", 224),
        "color_mode": kwargs.get("color_mode", "color"),
        "labeling_function": kwargs.get("model", ""),
        "num_robots": kwargs.get("num_robots", 48),
        "robot_ids": catalog_df["id"].values,
        "catalog_df": catalog_df,
    }

    robot_dataset = ConceptDataset(
        X=X,
        C=C,
        y=y,
        meta=meta,
        base_dir=results_dir / "robots",
    )

    return robot_dataset