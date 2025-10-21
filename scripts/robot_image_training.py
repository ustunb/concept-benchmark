import json, time, pickle
from pathlib import Path
import numpy as np
import torch
import copy
from torchvision import transforms
from concept_benchmark.models import ConceptDetector, FrontEndModel, RobotConceptClassifier
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot import create_synthetic_dataset
from scripts.dataset_skewing import create_skewed_splits, filter_training_by_string
from scripts.dnn_training import train_eval_image
from scripts.interventions import apply_interventions

# base_subconcepts = [
#         "foot_shape_flat_trapezoid",
#         "foot_shape_flat_rounded",
#         "foot_shape_flat_square",
#         "foot_shape_flat_5sided",
#         "foot_shape_flat_lshaped",
#         "foot_shape_pointy_trapezoid",
#         "foot_shape_pointy_rounded",
#         "foot_shape_pointy_square",
#         "foot_shape_pointy_3sided",
#         "foot_shape_pointy_4sided",
#     ]


settings = {
    "samples_per_instance": 3,
    "draw": 0,
    "CBM_type": "joint", #"sequential"
    "image_dir": "./data/robot_images",
    "image_size": "medium",
    "color_mode": "color",
    "train_dnn": 1,
    "seed": 555,
    "model": "'glorp' if (int(row['mouth_type']=='closed') +  int(row['foot_shape']=='pointy'))>= 2 else 'drent'",
    'dataset_characterization': "",
    "knows_concepts": False,
    "spurious_features": ["has_elbows", "hand_shape"],
    "drop_concepts": ["foot_shape_flat_rounded", "foot_shape_flat_5sided", 'foot_shape_flat_square',
                      'foot_shape_flat_trapezoid', "foot_shape_flat_lshaped", "foot_shape_flat_lshaped",
                      "foot_shape_pointy_trapezoid", "foot_shape_pointy_rounded", 'foot_shape_pointy_square',
                      'foot_shape_pointy_3sided', 'foot_shape_pointy_4sided'],
    "human_alignment": {"foot_shape": 1, "mouth_type": -1, "bias": -0.01}, # OR of ANDs model's logic
    "model_type": "stochastic",
    "logit_scalar": 4.0,
    "logit_intercept": 1.0,
    "label_noise_rate": 0,
    "missingness": "complete",
    "missing_rate": 1.0,
    "impute_missing": 0,
    "skew_concept": [
                     {'concepts': {'foot_shape_pointy_3sided': 1}, 'min_fraction': 0.25},
                     {'concepts': {'foot_shape_pointy_4sided': 1}, 'min_fraction': 0.25},
                     {'concepts': {'foot_shape_flat_5sided': 1}, 'min_fraction': 0.25},
                     {'concepts': {'foot_shape_flat_trapezoid': 1}, 'min_fraction': 0.25},
                     ], #[{'concepts': {'body_shape': 0, 'foot_shape': 1, 'has_antennae': 1}, 'min_fraction': 0.243},
                     # {'concepts': {'mouth_type': 0, 'foot_shape': 1, 'has_antennae': 1}, 'min_fraction': 0.2},
                     # {'concepts': {'body_shape': 0, 'mouth_type': 0, 'has_antennae': 1}, 'min_fraction': 0.15},
                     # {'concepts': {'body_shape': 1, 'mouth_type': 1, 'has_antennae': 0}, 'min_fraction': 0.235},
                     # {'concepts': {'body_shape': 1, 'foot_shape': 0, 'has_antennae': 0}, 'min_fraction': 0.2},
                     # {'concepts': {'foot_shape': 0, 'mouth_type': 1, 'has_antennae': 0}, 'min_fraction': 0.15}],#[{'concepts': {'mouth_type': 0, 'foot_shape_pointy_3sided': 1}, 'min_fraction': 0.13},
    "budget": [1,10],
    "intervention_accuracy": 0.9,
    "intervention_threshold": 0.1,
    "epochs": 10,
    "out_dir": str(results_dir / "robots"),
    "run_name": "special_CBM_DNN_f5_ft_p3_p4",
    "load_detector": "",#str(Path(results_dir / "robots" / "labeling_and_p3f4_medium_imbalanced3_rerun2" / "detector_dnn_robots_image_stochastic_complete__skewint-acc90_seed555.pt")),
    "load_frontend": "",#str(Path(results_dir / "robots" / "labeling_and_p3f4_medium_imbalanced3_rerun2" / "frontend_logreg_robots_image_stochastic_complete__skewint-acc90_seed555.pkl")),
}

# a script that takes the settings dictionary and changes it for subsequent runs; each run is for a different set of
# foot_shape subconcepts in the skew_concept dictionary (across all subsets of size at least 2) with others that are not
# in this dictionary, being in drop_concepts + foot_shape concept ; each run also has  unique name:
# name is: loop_footshape_<subconcepts_in_skew_only_first_letter_for_type_and_subtype>
def run_experiments_varying_footshape_subconcepts():
    from itertools import combinations
    base_subconcepts = [
        "foot_shape_flat_trapezoid",
        "foot_shape_flat_rounded",
        "foot_shape_flat_square",
        "foot_shape_flat_5sided",
        "foot_shape_flat_lshaped",
        "foot_shape_pointy_trapezoid",
        "foot_shape_pointy_rounded",
        "foot_shape_pointy_square",
        "foot_shape_pointy_3sided",
        "foot_shape_pointy_4sided",
    ]
    for r in range(2, len(base_subconcepts) + 1):
        for subset in combinations(base_subconcepts, r):
            if "pointy" not in "_".join(subset) or "flat" not in "_".join(subset):
                # skip subsets that do not have at least one pointy and one flat subtype
                continue
            S = copy.deepcopy(settings)
            skew_list = []
            drop_list = list(set(base_subconcepts) - set(subset))
            subset_pointy = [sc for sc in subset if "pointy" in sc]
            subset_flat = [sc for sc in subset if "flat" in sc]
            for sc in subset_pointy:
                skew_list.append({'concepts': {sc: 1}, 'min_fraction': round(0.5 / len(subset_pointy), 2)})
            for sc in subset_flat:
                skew_list.append({'concepts': {sc: 1}, 'min_fraction': round(0.5 / len(subset_flat), 2)})
            S["skew_concept"] = skew_list
            S["drop_concepts"] = drop_list + ["foot_shape"]
            S["run_name"] = "loop_footshape_" + "_".join([sc.split("_")[-2][0] + sc.split("_")[-1][0] for sc in subset])
            print(f"Running experiment with skewed subconcepts: {subset},mrun name: {S['run_name']}")
            main(S)

def _apply_missing(C, mode, rate, rng, y=None):
    if mode == "complete" or rate <= 0:
        return C
    C = C.copy().astype(np.float32)
    n, k = C.shape
    if mode == "mcar":
        M = rng.random((n, k)) < rate
    elif mode == "mar":
        if y is None:
            y = np.zeros(n, dtype=int)
        p1 = min(1.0, rate * 1.5)
        p0 = max(0.0, rate * 0.5)
        p = np.where(y.reshape(-1, 1) == 1, p1, p0)
        M = rng.random((n, k)) < p
    elif mode == "mnar":
        p = rate * (0.5 + 0.5 * C.astype(np.float32))
        M = rng.random((n, k)) < p
    else:
        M = np.zeros_like(C, dtype=bool)
    C[M] = -1.0
    return C


def _apply_label_noise(sample, noise_rate, seed):
    if noise_rate <= 0:
        return sample
    rng = np.random.default_rng(int(seed) + 4242)
    y = sample.y.astype(int).copy()
    flip_mask = rng.random(y.shape[0]) < float(noise_rate)
    y[flip_mask] = 1 - y[flip_mask]  # Flip labels

    return sample.__class__(
        parent=sample.parent, X=sample.X, C=sample.C, y=y, meta=sample.meta,
        transform=sample.transform, concept_transform=sample.concept_transform,
        target_transform=sample.target_transform, base_dir=getattr(sample, 'base_dir', None)
    )


def align_frontend_weights(frontend_model, concept_names, weight_dict):
    """
    Directly set frontend model weights for alignment.

    Args:
        frontend_model: Trained FrontEndModel instance
        concept_names: List of concept names (in training order)
        weight_dict: Dict mapping concept names to weights, plus 'bias' key
                    e.g. {'has_antennae': 1.0, 'body_shape': 1.0, 'bias': -2.0}

    Returns:
        Modified frontend model
    """
    lr_model = frontend_model.model

    n_concepts = len(concept_names)
    new_coef = np.zeros((1, n_concepts))

    for concept_name, weight in weight_dict.items():
        if concept_name == 'bias':
            continue
        if concept_name in concept_names:
            concept_idx = concept_names.index(concept_name)
            new_coef[0, concept_idx] = weight

    new_bias = weight_dict.get('bias', 0.0)

    lr_model.coef_ = new_coef
    lr_model.intercept_ = np.array([new_bias])

    return frontend_model


def _rate_tag(r):
    v = int(round(float(r) * 100))
    return f"{v:03d}"


def _get_foot_shape_pred(pred_row, concept_names):
    """Helper to extract foot_shape prediction consistently"""
    if 'foot_shape' in concept_names:
        return int(pred_row[concept_names.index('foot_shape')])

    # Check subtypes - if ANY pointy subtype is 1, return 1 (pointy), else 0 (flat)
    pointy_types = [c for c in concept_names if 'foot_shape_pointy' in c]
    for ptype in pointy_types:
        if pred_row[concept_names.index(ptype)] == 1:
            return 1  # pointy
    return 0  # flat


def test_concept_detector_invariance_point(concept_detector, concept, concept_names, point, dataset, device):
    """
    Test if the concept_detector for concept is invariant to changes it other features.

    Check that the concept prediction for one point will not change over other counterfactual points that share the same
    concept value.

    :param concept_detector:
    :param concept:
    :param concept_names:
    :param point:
    :param dataset:
    :param device:
    :return:
    """
    concept_idx = concept_names.index(concept)
    concept_value = dataset.C[point, concept_idx]
    if "foot_shape" in concept:
        foot_type = 1 if concept.split("_")[2] == "pointy" else 0
        foot_subtype = concept.split("_")[3]
    # Get all points that share the same concept value
    dataset_indices = dataset.meta["df_indices"]
    subcatalog = dataset.meta["catalog_df"].iloc[dataset_indices]
    matching_points = np.where(dataset.C[:, concept_idx] == concept_value)[0] if "foot_shape" not in concept else \
        np.where((subcatalog["foot_shape"] == foot_type) & (subcatalog["foot_shape_subtype"] == foot_subtype))[0]
    # Get the dataset with just the original point
    dataset_point = dataset.filter(np.array([i == point for i in range(len(dataset))]))
    original_pred = concept_detector.predict(dataset_point, embed_params={"device": device})[0, concept_idx]
    print("Original prediction for point", point, "concept", concept, "value", concept_value, "is", original_pred)
    # count the number of matching points that have different predictions
    variant_points = []
    dataset_mp = dataset.filter(np.array([i in matching_points for i in range(len(dataset))]))
    cf_preds = concept_detector.predict(dataset_mp, embed_params={"device": device})
    for i, mp in enumerate(matching_points):
        if mp == point:
            continue
        cf_pred = cf_preds[i, concept_idx]
        if (original_pred >= 0.5) != (cf_pred >= 0.5):
            print("Error on matching point", mp, "concept", concept, "value", concept_value, "prediction", cf_pred)
            variant_points.append(mp)

    if len(variant_points) > 0:
        print(f"Concept detector for concept '{concept}' is NOT invariant for point {point} with concept value {concept_value}.")
        return False
    return True


def test_concept_detector_invariance(concept_detector, concept_to_test, concept_names, dataset, device, num_tests=10):
    """
    Test concept detector invariance for all concepts on random points in the dataset.

    :param concept_detector:
    :param concept_to_test:
    :param concept_names:
    :param dataset:
    :param num_tests:
    :return:
    """
    rng = np.random.default_rng(12345)
    n_samples = len(dataset)
    all_passed = True
    for _ in range(num_tests):
        point = rng.integers(0, n_samples)
        passed = test_concept_detector_invariance_point(concept_detector, concept_to_test, concept_names, point, dataset, device)
        if not passed:
            all_passed = False
    return all_passed

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
    slug = f"robots_image_{model_type_tag}{miss_tag}{filter_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{impute_tag}"
    if S["run_name"]:
        run_dir = base_out / S["run_name"]
    else:
        run_dir = base_out / f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

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
                "round_circle",
                "round_oval",
                "round_oval2",
                "edgy_triangle",
                "edgy_square",
                "edgy_trapezoid",
            ],
            "foot_shape": [
                "flat_trapezoid",
                "flat_rounded",
                "flat_square",
                "flat_5sided",
                "flat_lshaped",
                "pointy_trapezoid",
                "pointy_rounded",
                "pointy_square",
                "pointy_3sided",
                "pointy_4sided",
            ],
        },
        "additional_features": [] if S.get("knows_concepts", True) else ["foot_shape_subtype"],
        "spurious_features": S.get("spurious_features", ["has_elbows", "hand_shape"]),
        "model": S.get("model", "'glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy')))>=1 else 'drent'"),
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
    }

    data = create_synthetic_dataset(**params)
    print(f"Current working directory: {Path.cwd()}")
    print(f"Dataset base_dir: {data._full.base_dir}")
    print(f"Image directory used in params: {params['output_directory']}")

    data.transform = tf
    data.generate_cvindices(seed=int(S["seed"]))

    if S.get("skew_concept") and S["skew_concept"]:
        # Use custom skewed splitting
        train, valid, test = create_skewed_splits(
            data,
            skew_specs=S["skew_concept"],
            rng=rng,
            drop_concepts=S.get("drop_concepts", [])
        )
    elif S.get("dataset_characterization", "") != "":
        train, valid, test = filter_training_by_string(
            data,
            string=S["dataset_characterization"],
            rng=rng
        )
    else:
        data.split("K05N01", fold_num_validation=4, fold_num_test=5)
        train = data.training; valid = data.validation; test = data.test

    # Setup the test set
    test_params = copy.deepcopy(params)
    standard_size = data.meta["num_unique_robots"]
    test_params["output_directory"] = Path(params['output_directory']) / "test_images"
    test_params["draw"] = True if not Path(test_params["output_directory"]).exists() or S.get("draw", False) else False
    test_params["samples_per_instance"] = int(params["test_set_size"] / standard_size) + 1
    test_data = create_synthetic_dataset(**test_params)
    test_data.drop_concepts(S.get("drop_concepts", []))
    test_data.transform = tf
    # take random sample of test set to match test_set_size
    rng_test = np.random.default_rng(int(S["seed"]) + 1234)
    test_indices = rng_test.choice(len(test_data), size=int(params["test_set_size"]), replace=False)
    test = test_data._full.filter(np.isin(np.arange(len(test_data)), test_indices))

    if S.get("label_noise_rate", 0.0) > 0:
        train = _apply_label_noise(train, S["label_noise_rate"], seed=int(S["seed"]))
        valid = _apply_label_noise(valid, S["label_noise_rate"], seed=int(S["seed"]))
        test = _apply_label_noise(test, S["label_noise_rate"], seed=int(S["seed"]))

    if miss != "complete" and rate > 0:
        Ctr = _apply_missing(train.C, miss, rate, rng, y=train.y.astype(int))
        train = train.__class__(parent=train.parent, X=train.X, C=Ctr, y=train.y, meta=train.meta, transform=train.transform, concept_transform=train.concept_transform, target_transform=train.target_transform, base_dir=train.base_dir)

    # print a breakdown of unique robots per each fot shape subtype in the training set
    # print the proportion of each unique robot in the training dataset by foot shape subtype
    # print("Training set unique robot distribution by foot shape subtype:")
    # foot_shape_concept_names = [c for c in train.concepts if 'foot_shape' in c]
    # foot_shape_concept_indices = [train.concepts.index(c) for c in foot_shape_concept_names]
    # unique_robots = {}
    # for i in range(len(train.C)):
    #     robot_key = tuple(train.C[i, :])
    #     if robot_key not in unique_robots:
    #         unique_robots[robot_key] = 0
    #     unique_robots[robot_key] += 1
    # # for each subtype enumerate all unique robots and for each unique robot count how many have that subtype
    # subtype_counts = {c: {r: 0 for r in unique_robots.keys()} for c in foot_shape_concept_names}
    # # iterate over the whole training set to count
    # for i in range(len(train.C)):
    #     robot_key = tuple(train.C[i, :])
    #     for c, c_idx in zip(foot_shape_concept_names, foot_shape_concept_indices):
    #         if train.C[i, c_idx] == 1:
    #             subtype_counts[c][robot_key] += 1
    # for c in foot_shape_concept_names:
    #     total = sum(subtype_counts[c].values())
    #     print(f"  {c}:")
    #     for robot_key, count in subtype_counts[c].items():
    #         if count > 0:
    #             print(f"    Robot {robot_key}: {count} ({count/total:.1%})")


    # print distribution of each concept in the training set
    print("Training set concept distributions:")
    for i, concept_name in enumerate(train.concepts):
        unique, counts = np.unique(train.C[:, i], return_counts=True)
        dist = dict(zip(unique, counts))
        total = counts.sum()
        dist_str = ", ".join([f"{int(k)}: {v} ({v/total:.1%})" for k, v in dist.items()])
        print(f"  {concept_name}: {dist_str}")

    # print distribution of each concept in the test set
    print("Test set concept distributions:")
    for i, concept_name in enumerate(test.concepts):
        unique, counts = np.unique(test.C[:, i], return_counts=True)
        dist = dict(zip(unique, counts))
        total = counts.sum()
        dist_str = ", ".join([f"{int(k)}: {v} ({v/total:.1%})" for k, v in dist.items()])
        print(f"  {concept_name}: {dist_str}")

    # print num glorps and dreints in the training vs test set:
    def count_classes(sample):
        y = sample.y.astype(int)
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))
        total = counts.sum()
        dist_str = ", ".join([f"{int(k)}: {v} ({v/total:.1%})" for k, v in dist.items()])
        return dist_str
    print("Training set class distribution:", count_classes(train))
    print("Test set class distribution:", count_classes(test))


    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    config = {
        'device': device,
        'batch_size': 32,
        'num_workers': 0 if device == 'mps' else 12,
        'pin_memory': False if device == 'mps' else True,
    }
    cd = ConceptDetector(model=RobotConceptClassifier(num_concepts=train.n_concepts, input_size=600 if S["image_size"] == "large" else 32 if S["image_size"] == "medium" else 8))
    det_name = f"detector_dnn_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"
    if S["load_detector"]:
        mini_train = train.filter(np.array([True] + [False] * (len(train.C) - 1)))
        mini_valid = valid.filter(np.array([True] + [False] * (len(valid.C) - 1)))
        cd.fit(mini_train, mini_valid, freeze=True, embed_params={"device": device}, fit_params={"epochs": 1, "device": "cpu"})
        state = torch.load(S["load_detector"], weights_only=False, map_location="cpu")
        cd.load_state_dict(state)
        det_path = Path(S["load_detector"])
    else:
        cd.fit(train, valid, embed_params={'shuffle': False, **config}, fit_params={"epochs": 50, 'lr': 1e-3, "patience": 10, **config})
        det_path = run_dir / det_name
        torch.save(cd.state_dict(), det_path)

    if int(S.get('fe_harness', 0)) == 1:
        from scripts.fe_harness import run_fe_harness
        names = list(train.concepts)
        C_tr = train.C.astype(float)
        C_te = test.C.astype(float)
        y_tr = train.y.astype(int)
        y_te = test.y.astype(int)
        try:
            P_tr = cd.predict_proba(train)
            P_te = cd.predict_proba(test)
        except Exception:
            P_tr = cd.predict(train).astype(float)
            P_te = cd.predict(test).astype(float)
        run_fe_harness(C_tr, y_tr, C_te, y_te, P_tr, P_te, names, table_name="FE 2x2 (coarse vs sub; C vs P)")

    # test invariance of concept detectors
    subtype_concepts = [c for c in test.concepts if 'foot_shape_' in c]
    for concept in subtype_concepts:
        invariance_passed = test_concept_detector_invariance(cd, concept, train.concepts, test, device, num_tests=10)
        print (f"Invariance test for concept '{concept}': {'PASSED' if invariance_passed else 'FAILED'}")


    P_tr = cd.predict(train, embed_params={"device": device})
    P_vl = cd.predict(valid, embed_params={"device": device})
    P_te = cd.predict(test, embed_params={"device": device})
    H_tr = (P_tr > 0.5).astype(np.float32)
    H_te = (P_te > 0.5).astype(np.float32)
    H_vl = (P_vl > 0.5).astype(np.float32)

    # get concept accuracy
    concept_names = test.concepts
    per_concept_acc = {}
    train_per_concept_acc = {}
    for i, concept_name in enumerate(concept_names):
        true_labels = test.C[:, i]
        train_true_labels = train.C[:, i]
        train_labels = H_tr[:, i]
        pred_labels = H_te[:, i]

        accuracy = float((pred_labels == true_labels).mean())
        train_accuracy = float((train_labels == train_true_labels).mean())
        train_per_concept_acc[concept_name] = train_accuracy
        per_concept_acc[concept_name] = accuracy
        print(pred_labels)
        print(f"{concept_name}: {accuracy:.4f}")

    fe = FrontEndModel()
    fe_name = f"frontend_logreg_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pkl"
    if S["load_frontend"]:
        with open(S["load_frontend"], "rb") as f:
            fe = pickle.load(f)
        fe_path = Path(S["load_frontend"])
    else:
        Ctr = train.C.astype(np.float32)
        if int(S["impute_missing"]) and np.any(Ctr < 0):
            Cin = Ctr.copy()
            m = Cin < 0
            Cin[m] = P_tr[m]
            fe.fit(Cin, train.y.astype(int))
        else:
            if S.get("CBM_type", "joint") == "sequential":
                keep = np.all(Ctr >= 0, axis=1)
                fe.fit(H_tr[keep], train.y[keep].astype(int))
            else:
                fe.fit(Ctr, train.y.astype(int))
        fe_path = run_dir / fe_name
        with open(fe_path, "wb") as f:
            pickle.dump(fe, f)

    y_pred_det = fe.predict_proba(H_te)
    y_pred_gt = fe.predict_proba(test.C.astype(np.float32))
    acc_det = float((y_pred_det.argmax(1) == test.y.astype(int)).mean())
    acc_gt = float((y_pred_gt.argmax(1) == test.y.astype(int)).mean())
    concept_acc_mean = float((H_te == test.C).mean())


    print("\n=== Learned Frontend Weights ===")
    for i, concept in enumerate(test.concepts):
        print(f"  {concept}: {fe.model.coef_[0, i]:.4f}")
    print(f"  bias: {fe.model.intercept_[0]:.4f}")


    import pandas as pd
    for split_name, split_data, split_preds in [("train", train, fe.predict_proba(H_tr)),
                                                ("validation", valid, fe.predict_proba(H_vl)),
                                                ("test", test, y_pred_det)]:
        rows = []
        print("Accuracy on", split_name, "set:", float((split_preds.argmax(1) == split_data.y.astype(int)).mean()))
        for i in range(len(split_data.C)):
            pred_label = int(split_preds[i].argmax())
            true_label = int(split_data.y[i])
            row_data = {
                'sample_idx': i,
                'body_shape_pred': int(H_tr[i, train.concepts.index('body_shape')]) if split_name == "train" else\
                    int(H_te[i, test.concepts.index('body_shape')]) if split_name == "test" else int(H_vl[i, valid.concepts.index('body_shape')]),
                'mouth_type_pred': int(H_tr[i, train.concepts.index('mouth_type')]) if split_name == "train" else\
                    int(H_te[i, test.concepts.index('mouth_type')]) if split_name == "test" else int(H_vl[i, valid.concepts.index('mouth_type')]),
                'foot_shape_pred': _get_foot_shape_pred(
                    H_tr[i] if split_name == "train" else H_te[i] if split_name == "test" else H_vl[i],
                    train.concepts if split_name == "train" else test.concepts if split_name == "test" else valid.concepts
                ),
                # Ground truth from UC
                'body_shape': int(split_data.meta['UC'][i, split_data.meta['unfiltered_concepts'].index('body_shape')]),
                'mouth_type': int(split_data.meta['UC'][i, split_data.meta['unfiltered_concepts'].index('mouth_type')]),
                'foot_shape': int(split_data.meta['UC'][i, split_data.meta['unfiltered_concepts'].index('foot_shape')]),
                "foot_shape_subtype_string": split_data.meta['catalog_df'].iloc[split_data.meta['df_indices'][i]][
                    'foot_shape_subtype'],
                'predicted': pred_label,
                'true_label': true_label,
            }
            rows.append(row_data)
        df = pd.DataFrame(rows)
        print(f"\nSample of {split_name} set predictions:")
        print(df.head(100).to_string())


    # BASELINE
    dnn_stats = {}
    if S.get("train_dnn", False):
        print("Training baseline DNN...")

        # Convert ConceptDatasetSample to path arrays
        paths_tr = [train.base_dir / p for p in train.X]
        ytr = train.y.astype(int)
        paths_te = [test.base_dir / p for p in test.X]
        yte = test.y.astype(int)

        dnn_acc, proc, dnn_model = train_eval_image(
            paths_tr, ytr, paths_te, yte,
            model_id=S.get("image_model", "google/vit-base-patch16-224"),
            epochs=int(S["epochs"]),
            batch_size=16,
            lr=5e-5,
            device=device
        )

        dnn_stats = {"dnn_accuracy": float(dnn_acc)}
        print(f"DNN accuracy: {float(dnn_acc)}")

        dnn_name = f"dnn_vitb16_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"
        dnn_path = run_dir / dnn_name
        torch.save({
            "model_state_dict": dnn_model.state_dict(),
            "processor": proc,
        }, dnn_path)


    # INTERVENTIONS
    intervention_results = {}
    budgets = S.get('budget', [1, 2, 3, 4, 5])
    human_acc = S.get("intervention_accuracy", 1.0)
    for budget in budgets:
        for policy in ["top-1", "top-k"]:
            H_intervened, intervention_stats = apply_interventions(
                pred_probs=P_te,
                ground_truth=test.C.astype(int),
                frontend_model=fe,
                budget_k=budget,
                intervention_threshold=S.get("intervention_threshold", 0.5),
                human_accuracy=human_acc,
                policy=policy,
                rng=rng
            )

            # Calculate accuracy after interventions
            y_pred_intervened = fe.predict_proba(H_intervened)
            acc_intervened = float((y_pred_intervened.argmax(1) == test.y.astype(int)).mean())

            # Store results
            key = f"budget_{budget}_{policy}_human_acc_{int(human_acc * 100)}"
            intervention_results[key] = {
                "accuracy": acc_intervened,
                "accuracy_gain": acc_intervened - acc_det,
                "predictions_intervened_on": intervention_stats["samples_intervened_on"],
                "interventions_rate": intervention_stats["intervention_rate"],
                "avg_edits_per_intervention": intervention_stats["avg_edits_per_intervention"],
                "total_concept_checks": intervention_stats["total_concept_checks"],
                "total_concept_edits_made": intervention_stats["total_concept_edits_made"],
                "policy": policy,
                "budget": budget,
                "human_accuracy": human_acc
            }

    # ANALYSIS
    all_preds = []
    original_probs = fe.predict_proba(H_te)
    original_preds = original_probs.argmax(1)
    for i in range(len(test.y)):
        true_label = int(test.y[i])
        pred_label = int(original_preds[i])

        row_data = {
            'sample_idx': i,
            # Ground truth from UC
            'foot_shape': int(test.meta['UC'][i, test.meta['unfiltered_concepts'].index('foot_shape')]),
            "foot_shape_subtype_string": test.meta['catalog_df'].iloc[test.meta['df_indices'][i]]['foot_shape_subtype'],
            'predicted': pred_label,
            'true_label': true_label,
        }
        # get what each detector predicts for this case:
        for j, concept in enumerate(test.concepts):
            row_data[f"{concept}_pred"] = int(float(P_te[i, j]) > 0.5)
            row_data[f"{concept}"] = int(test.C[i, j])
        all_preds.append(row_data)

    # for the existing subtype detectors (.startswith(foot_shape_pointy) or .startswith(foot_shape_flat) in test.concepts)
    # check how often in predicted 1 when the subtype string was any of the subconcepts in drop_concepts (again, .startsiwth)
    # store as percentage of total cases where that subconcept was present
    subtype_concepts = [c for c in test.concepts if c.startswith('foot_shape_')]
    missing_concepts = [c for c in S.get("drop_concepts", []) if c.startswith('foot_shape_')]
    all_preds = pd.DataFrame(all_preds)
    predicted_classes = subtype_concepts + ['other']
    all_concepts = sorted(subtype_concepts + missing_concepts)
    all_concepts = [c for c in all_concepts if "foot_shape_" in c]

    confusion_matrix = {true_subtype: {pred_class: 0 for pred_class in predicted_classes}
                        for true_subtype in all_concepts}

    for idx, row in all_preds.iterrows():
        true_subtype = row['foot_shape_subtype_string']
        foot_shape = "pointy" if row['foot_shape'] else "flat"
        true_type = f"foot_shape_{foot_shape}_{true_subtype}"

        # Find which detector(s) fired (predicted subtype)
        fired_detectors = [det for det in subtype_concepts if row[f"{det}_pred"] == 1]

        if len(fired_detectors) == 0:
            # No detector fired -> predict "other"
            fired_detectors = ['other']
        for ps in fired_detectors:
            confusion_matrix[true_type][ps] += 1

    # make a pd dataframe for better visualization
    confusion_df = pd.DataFrame(confusion_matrix).T
    # print
    print("\nConfusion Matrix for Foot Shape Subtype Detectors:")
    print(confusion_df.to_string())

    # get model accuracy for each subtype concept:
    per_concept_acc2 = {}
    for concept in sorted(subtype_concepts + missing_concepts):
        foot_type = "pointy" if "pointy" in concept else "flat"
        foot_subtype = concept.replace('foot_shape_', '').replace(foot_type + "_", "")
        concept_rows = all_preds[(all_preds['foot_shape_subtype_string'] == foot_subtype) & (all_preds['foot_shape'] == (1 if foot_type == "pointy" else 0))]
        if len(concept_rows) > 0:
            accuracy = float((concept_rows['predicted'] == concept_rows['true_label']).mean())
            per_concept_acc2[concept] = round(accuracy, 4)
        else:
            per_concept_acc2[concept] = None

    print(per_concept_acc2)


    # ALIGNMENT
    alignment_stats = {}
    if S.get("human_alignment", {}) != {}:
        test_concepts = H_te
        test_labels = test.y.astype(int)
        original_frontend = fe
        aligned_frontend = copy.deepcopy(fe)
        aligned_frontend = align_frontend_weights(aligned_frontend, test.concepts, S.get("human_alignment", {}))

        original_probs = original_frontend.predict_proba(test_concepts)
        aligned_probs = aligned_frontend.predict_proba(test_concepts)
        original_preds = original_probs.argmax(1)
        aligned_preds = aligned_probs.argmax(1)

        original_acc = (original_preds == test_labels).mean()
        aligned_acc = (aligned_preds == test_labels).mean()

        print("\n=== Aligned Frontend Weights ===")
        for i, concept in enumerate(test.concepts):
            print(f"  {concept}: {aligned_frontend.model.coef_[0, i]:.4f}")
        print(f"  bias: {aligned_frontend.model.intercept_[0]:.4f}")

        # Use the model's predict_proba method (returns probabilities for both classes)
        aligned_probs_both = aligned_frontend.predict_proba(H_te)  # Shape: (n_samples, 2)
        aligned_probs_glorp = aligned_probs_both[:, 1]  # Probability of class 1 (Glorp)

        # Use the model's predict method (applies >= 0.5 threshold internally)
        aligned_preds = aligned_frontend.predict(H_te)

        alignment_stats = {
            'original_accuracy': float(original_acc),
            'aligned_accuracy': float(aligned_acc),
            'accuracy_change': float(aligned_acc - original_acc),
            'predictions_changed': int(np.sum(original_preds != aligned_preds))
        }

    meta_name = f"meta_cbm_detected_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.json"
    metrics_name = f"metrics_cbm_detected_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.json"

    meta = {
        "settings": S,
        "run_dir": str(run_dir),
        "artifacts": {
            "detector": str(det_path),
            "frontend": str(fe_path),
        },
        "splits": {
            "n_train": int(train.n),
            "n_valid": int(valid.n),
            "n_test": int(test.n),
        },
        "concepts": list(data.concepts) if hasattr(data, "concepts") else [],
        "intervention_budgets": budgets,
        "intervention_acc": human_acc,
        "naming_slug": slug,
    }

    metrics = {
        "cbm_acc_detected": acc_det,
        "cbm_acc_oracle": acc_gt,
        "concept_det_acc_mean": concept_acc_mean,
        "interventions": intervention_results,
    }
    metrics.update(dnn_stats)
    metrics.update({"alignment": alignment_stats})
    feweights = {}
    for i, concept in enumerate(test.concepts):
        feweights[concept] = round(fe.model.coef_[0, i], 4)
    feweights["bias"] = round(fe.model.intercept_[0], 4)
    metrics["frontend_weights"] = feweights
    metrics.update({'concept_accuracies': per_concept_acc,
                    "model_accuracies_per_concept": per_concept_acc2, 'train_concept_accuracies': train_per_concept_acc})

    meta_path = run_dir / meta_name
    metrics_path = run_dir / metrics_name
    confusion_path = run_dir / "confusion.csv"
    confusion_df.to_csv(confusion_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({
        "meta_path": str(meta_path),
        "metrics_path": str(metrics_path),
        "detector_path": str(det_path),
        "frontend_path": str(fe_path),
    }, indent=2))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fe-harness', dest='fe_harness', type=int, default=0)
parser.add_argument('--draw', dest='draw', type=int)
parser.add_argument('--image-size', dest='image_size', type=str)
parser.add_argument('--train-dnn', dest='train_dnn', type=int)
parser.add_argument('--model', dest='model', type=str)
parser.add_argument('--drop-concepts', dest='drop_concepts', type=str)          # JSON list
parser.add_argument('--model-type', dest='model_type', type=str)
parser.add_argument('--logit-scalar', dest='logit_scalar', type=float)
parser.add_argument('--logit-intercept', dest='logit_intercept', type=float)
parser.add_argument('--logit-weights', dest='logit_weights', type=str)          # JSON dict
parser.add_argument('--skew-concept', dest='skew_concept', type=str)            # JSON list[dict]
parser.add_argument('--run-name', dest='run_name', type=str)

args, _ = parser.parse_known_args()

overrides = {k: v for k, v in vars(args).items() if v is not None}

# Parse JSON-like args
if 'drop_concepts' in overrides:
    overrides['drop_concepts'] = json.loads(overrides['drop_concepts'])
if 'logit_weights' in overrides:
    overrides['logit_weights'] = json.loads(overrides['logit_weights'])
if 'skew_concept' in overrides:
    overrides['skew_concept'] = json.loads(overrides['skew_concept'])

settings.update(overrides)


main(settings)

#run_experiments_varying_footshape_subconcepts()
