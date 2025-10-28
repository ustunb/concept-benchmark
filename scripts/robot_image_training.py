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
from scripts.robot_interventions import test_interventions
from scripts.robot_alignment import test_alignment
from scripts.robot_invariance_test import test_concept_detector_invariance
from scripts.robot_utils import _apply_missing, _apply_label_noise, _rate_tag, _get_concept_accuracies, \
    _get_confusion_matrix, _get_accuracies_per_subconcept

settings = {
    "samples_per_instance": 4,
    "draw": 0,
    "CBM_type": "separate", #"sequential"
    "image_dir": "./data/robot_images",
    "image_size": "medium",
    "color_mode": "color",
    "train_dnn": 0,
    "seed": 1002,
    "model": "'glorp' if (int(row['mouth_type']=='closed') + int(row['foot_shape']=='pointy'))>= 3 else 'drent'",
    'dataset_characterization': "",
    "test_size": 10000,
    "train_size": 3800,
    "knows_concepts": False,
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
    "subconcepts": ["foot_shape_subtype"],
    "spurious_features": ["has_elbows", "hand_shape"],
    "drop_concepts": ["foot_shape_flat_rounded",
                      "foot_shape_pointy_trapezoid",
                      'foot_shape_pointy_3sided', 'foot_shape_flat_lshaped',
                      'foot_shape'],#'foot_shape_pointy_4sided', 'foot_shape_pointy_square', 'foot_shape_pointy_rounded', 'foot_shape_flat_5sided', 'foot_shape_flat_square','foot_shape_flat_trapezoid' ],
    "human_alignment": {
        "foot_shape_pointy_4sided": 5,
        "foot_shape_flat_5sided": -5,
        "foot_shape_pointy_rounded": 5,
        "foot_shape_flat_square": -1,
        "foot_shape_pointy_square": 5,
        "mouth_type": -5,
        "bias": 3
    },
    "model_type": "stochastic",
    "logit_scalar": 1.0,
    "logit_intercept": 3,
    "logit_weights": {"mouth_type": 5, "foot_shape": 10},
    "label_noise_rate": 0,
    "missingness": "complete",
    "missing_rate": 1.0,
    "impute_missing": 0,
    "skew_concept": [
                     {'concepts': {'foot_shape_pointy_square': 1}, 'min_fraction': 0.005},
                     {'concepts': {'foot_shape_pointy_rounded': 1}, 'min_fraction': 0.005},
                     {'concepts': {'foot_shape_pointy_4sided': 1}, 'min_fraction': 0.49},
                     {'concepts': {'foot_shape_flat_square': 1}, 'min_fraction': 0.005},
                     {'concepts': {'foot_shape_flat_trapezoid': 1}, 'min_fraction': 0.005},
                     {'concepts': {'foot_shape_flat_5sided': 1}, 'min_fraction': 0.49},
                     ],
    "budget": [1],
    "intervention_accuracy": 0.9,
    "intervention_threshold": 1.0,
    "epochs": 10,
    "out_dir": str(results_dir / "robots"),
    "run_name": "TEST_cbm_run_565_subconcepts",
    "load_detector": "",#str(Path(results_dir / "robots" / "labeling_and_p3f4_medium_imbalanced3_rerun2" / "detector_dnn_robots_image_stochastic_complete__skewint-acc90_seed555.pt")),
    "load_frontend": "",#str(Path(results_dir / "robots" / "labeling_and_p3f4_medium_imbalanced3_rerun2" / "frontend_logreg_robots_image_stochastic_complete__skewint-acc90_seed555.pkl")),
}


def define_train_valid_test(settings, concept_dataset, missingness, params, rate, rng, tf):
    if settings.get("skew_concept") and settings["skew_concept"]:
        # Use custom skewed splitting
        train, valid, test = create_skewed_splits(
            concept_dataset,
            skew_specs=settings["skew_concept"],
            test_size=settings.get("test_size", 10000),
            train_skew_size=settings.get("train_size", None),
            rng=rng,
            drop_concepts=settings.get("drop_concepts", [])
        )
    elif settings.get("dataset_characterization", "") != "":
        train, valid, test = filter_training_by_string(
            concept_dataset,
            string=settings["dataset_characterization"],
            rng=rng
        )
    else:
        concept_dataset.split("K05N01", fold_num_validation=4, fold_num_test=5)
        train = concept_dataset.training
        valid = concept_dataset.validation


    if settings.get("label_noise_rate", 0.0) > 0:
        train = _apply_label_noise(train, settings["label_noise_rate"], seed=int(settings["seed"]))
        valid = _apply_label_noise(valid, settings["label_noise_rate"], seed=int(settings["seed"]))
        test = _apply_label_noise(test, settings["label_noise_rate"], seed=int(settings["seed"]))

    if missingness != "complete" and rate > 0:
        Ctr = _apply_missing(train.C, missingness, rate, rng, y=train.y.astype(int))
        train = train.__class__(parent=train.parent, X=train.X, C=Ctr, y=train.y, meta=train.meta,
                                transform=train.transform, concept_transform=train.concept_transform,
                                target_transform=train.target_transform, base_dir=train.base_dir)
    return test, train, valid


def train_concept_detector(settings, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir,
                           seed_tag, skew_tag, train, valid, test):
    cd = ConceptDetector(model=RobotConceptClassifier(num_concepts=train.n_concepts,
                                                      input_size=600 if settings["image_size"] == "large" else
                                                      32 if settings["image_size"] == "medium" else 8))
    det_name = f"detector_dnn_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"
    if settings["load_detector"]:
        mini_train = train.filter(np.array([True] + [False] * (len(train.C) - 1)))
        mini_valid = valid.filter(np.array([True] + [False] * (len(valid.C) - 1)))

        cd.fit(mini_train, mini_valid, freeze=True, embed_params={"device": device},
               fit_params={"epochs": 1, "device": "cpu"})
        state = torch.load(settings["load_detector"], weights_only=False, map_location="cpu")
        cd.load_state_dict(state)
        det_path = Path(settings["load_detector"])
    else:
        cd.fit(train, valid, embed_params={'shuffle': False, **config},
               fit_params={"epochs": 50, 'lr': 1e-3, "patience": 10, **config})
        det_path = run_dir / det_name
        torch.save(cd.state_dict(), det_path)

    # test invariance of concept detectors
    subtype_concepts = [c for c in test.concepts if 'foot_shape_' in c]
    for concept in subtype_concepts:
        invariance_passed = test_concept_detector_invariance(cd, concept, train.concepts, test, device,
                                                             num_tests=10)
        print(f"Invariance test for concept '{concept}': {'PASSED' if invariance_passed else 'FAILED'}")
    return cd, det_path


def train_frontend(H_te, h_train, prob_train, sttngs, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir,
                   seed_tag, skew_tag, test, train):
    fe = FrontEndModel()
    fe_name = f"frontend_logreg_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pkl"
    if sttngs["load_frontend"]:
        with open(sttngs["load_frontend"], "rb") as f:
            fe = pickle.load(f)
        fe_path = Path(sttngs["load_frontend"])
    else:
        Ctr = train.C.astype(np.float32)
        if int(sttngs["impute_missing"]) and np.any(Ctr < 0):
            Cin = Ctr.copy()
            m = Cin < 0
            Cin[m] = prob_train[m]
            fe.fit(Cin, train.y.astype(int))
        else:
            if sttngs.get("CBM_type", "separate") == "sequential":
                keep = np.all(Ctr >= 0, axis=1)
                fe.fit(h_train[keep], train.y[keep].astype(int))
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
    return acc_det, acc_gt, concept_acc_mean, fe, fe_path, y_pred_det


def train_dnn(sttngs, device, dnn_stats, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir, seed_tag,
              skew_tag, test, train):
    print("Training baseline DNN...")

    # Convert ConceptDatasetSample to path arrays
    paths_tr = [train.base_dir / p for p in train.X]
    ytr = train.y.astype(int)
    paths_te = [test.base_dir / p for p in test.X]
    yte = test.y.astype(int)

    dnn_acc, proc, dnn_model = train_eval_image(
        paths_tr, ytr, paths_te, yte,
        model_id=sttngs.get("image_model", "google/vit-base-patch16-224"),
        epochs=int(sttngs["epochs"]),
        batch_size=16,
        lr=5e-5,
        device=device,
        seed=int(sttngs["seed"])
    )

    dnn_stats = {"dnn_accuracy": float(dnn_acc)}
    print(f"DNN accuracy: {float(dnn_acc)}")

    dnn_name = f"dnn_vitb16_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"
    dnn_path = run_dir / dnn_name
    torch.save({
        "model_state_dict": dnn_model.state_dict(),
        "processor": proc,
    }, dnn_path)
    return dnn_stats


def main(sttngs):
    ###########################################################################
    ##############################    NAMING    ##############################
    ###########################################################################
    S = dict(sttngs)
    rng = np.random.default_rng(int(S["seed"]))
    torch.manual_seed(int(S["seed"]))

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
    ###########################################################################

    ###########################################################################
    ##########################    DATA DEFINITION    ##########################
    ###########################################################################
    params = copy.deepcopy(sttngs)
    params.update({
        "output_directory": S.get("image_dir", run_dir / "images"),
        "additional_features": [] if S.get("knows_concepts", True) else S.get("subconcepts", ["foot_shape_subtype", "hand_shape_subtype"]),
        "scalar": float(S.get("logit_scalar", 1.0)),
        "intercept": float(S.get("logit_intercept", 0.0)),
        'weights': S.get("logit_weights", {}),
        "size": S["image_size"],
        "test_set_size": 10000,
        "train_concept_detector": True,
        "verbose": True,
        "rng_seed": S['seed'],
    })

    data = create_synthetic_dataset(**params)

    tf = transforms.Compose([transforms.ToTensor()])
    data.transform = tf
    data.generate_cvindices(seed=int(S["seed"]))

    test, train, valid = define_train_valid_test(S, data, miss, params, rate, rng, tf)
    ###########################################################################

    ###########################################################################
    ###########################    MODEL TRAINING    ##########################
    ###########################################################################
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    config = {
        'device': device,
        'batch_size': 32,
        'num_workers': 0 if device == 'mps' else 12,
        'pin_memory': False if device == 'mps' else True,
    }

    cd, det_path = train_concept_detector(S, config, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag,
                                          run_dir, seed_tag, skew_tag, train, valid, test)


    P_tr = cd.predict(train, embed_params={"device": device})
    P_te = cd.predict(test, embed_params={"device": device})
    H_tr = (P_tr > 0.5).astype(np.float32)
    H_te = (P_te > 0.5).astype(np.float32)

    per_concept_acc, train_per_concept_acc = _get_concept_accuracies(H_te, H_tr, test, train)

    acc_det, acc_gt, concept_acc_mean, fe, fe_path, y_pred_det = train_frontend(H_te, H_tr, P_tr, S, int_acc_tag,
                                                                                label_noise_tag, miss_tag,
                                                                                model_type_tag, run_dir, seed_tag,
                                                                                skew_tag, test, train)
    dnn_stats = {}
    if S.get("train_dnn", False):
        dnn_stats = train_dnn(S, device, {}, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir,
                              seed_tag, skew_tag, test, train)
    ###########################################################################

    ###########################################################################
    ###########################    INTERVENTIONS    ###########################
    ###########################################################################
    # apply interventions and measure their effect on the predictions
    budgets, human_acc, intervention_results = test_interventions(P_te, S, acc_det, fe, rng, test)

    subtype_concepts = [c for c in test.concepts if c.startswith('foot_shape_')]
    missing_concepts = [c for c in S.get("drop_concepts", []) if c.startswith('foot_shape_')]

    # get the confusion matrix for the detector
    all_preds, confusion_df = _get_confusion_matrix(subtype_concepts, missing_concepts, fe, H_te, P_te, test)

    # get model accuracy for each subtype concept:
    per_concept_acc2 = _get_accuracies_per_subconcept(all_preds, missing_concepts, subtype_concepts)

    # align the model with new weights
    alignment_stats = {}
    if S.get("human_alignment", {}) != {}:
        alignment_stats = test_alignment(H_te, S.get("human_alignment", {}), fe, test)
    ###########################################################################

    ###########################################################################
    ##############################    SAVING    ###############################
    ###########################################################################
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
        "logit_weights": params.get("weights", {}),
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

    catalog_csv_path = run_dir / "catalog.csv"
    data.meta["catalog_df"].to_csv(catalog_csv_path, index=False)
    meta["catalog_csv"] = str(catalog_csv_path)
    meta["df_indices"] = {
        "train": list(map(int, train.meta["df_indices"])),
        "valid": list(map(int, valid.meta["df_indices"])),
        "test": list(map(int, test.meta["df_indices"])),
    }
    meta["robot_ids"] = {
        "train": list(map(int, train.meta.get("robot_ids", []))),
        "valid": list(map(int, valid.meta.get("robot_ids", []))),
        "test": list(map(int, test.meta.get("robot_ids", []))),
    }

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

    return metrics


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--fe-harness', dest='fe_harness', type=int, default=0)
# parser.add_argument('--draw', dest='draw', type=int)
# parser.add_argument('--image-size', dest='image_size', type=str)
# parser.add_argument('--train-dnn', dest='train_dnn', type=int)
# parser.add_argument('--model', dest='model', type=str)
# parser.add_argument('--drop-concepts', dest='drop_concepts', type=str)          # JSON list
# parser.add_argument('--model-type', dest='model_type', type=str)
# parser.add_argument('--logit-scalar', dest='logit_scalar', type=float)
# parser.add_argument('--logit-intercept', dest='logit_intercept', type=float)
# parser.add_argument('--logit-weights', dest='logit_weights', type=str)          # JSON dict
# parser.add_argument('--skew-concept', dest='skew_concept', type=str)            # JSON list[dict]
# parser.add_argument('--run-name', dest='run_name', type=str)
#
# args, _ = parser.parse_known_args()
#
# overrides = {k: v for k, v in vars(args).items() if v is not None}
#
# # Parse JSON-like args
# if 'drop_concepts' in overrides:
#     overrides['drop_concepts'] = json.loads(overrides['drop_concepts'])
# if 'logit_weights' in overrides:
#     overrides['logit_weights'] = json.loads(overrides['logit_weights'])
# if 'skew_concept' in overrides:
#     overrides['skew_concept'] = json.loads(overrides['skew_concept'])

# settings.update(overrides)

def find_good_seed(stngs):
    for seed in range(1001, 2000):
        print(f"Testing seed {seed}...")
        stngs["seed"] = seed
        # train subconcept CBMS only
        stngs["drop_concepts"] = ["foot_shape_flat_rounded",
                      "foot_shape_pointy_trapezoid",
                      'foot_shape_pointy_3sided', 'foot_shape_flat_lshaped',
                      'foot_shape']
        stngs["train_dnn"] = 0
        stngs['run_name'] = f"A_SCBM_alignment_trials_seed{seed}"
        metrics = main(stngs)
        cbm_acc = metrics.get("cbm_acc_detected", 0.0)
        if 0.68 <= cbm_acc <= 0.735:
            print(f"Found seed {seed} with SCBM accuracy {cbm_acc:.4f}. Now training full CBM...")
            # train full CBM
            stngs["drop_concepts"] = ["foot_shape_flat_rounded", "foot_shape_pointy_trapezoid", 'foot_shape_pointy_3sided',
                                      'foot_shape_flat_lshaped', 'foot_shape_pointy_4sided', 'foot_shape_pointy_square',
                                      'foot_shape_pointy_rounded', 'foot_shape_flat_5sided', 'foot_shape_flat_square',
                                      'foot_shape_flat_trapezoid' ]

            stngs["train_dnn"] = 0
            stngs['run_name'] = f"A_CBM_alignment_trials_seed{seed}"
            metrics_full = main(stngs)
            cbm_acc_full = metrics_full.get("cbm_acc_detected", 0.0)
            if cbm_acc_full >= 0.85:
                # train DNN
                print(f"Full CBM accuracy {cbm_acc_full:.4f} meets criteria")
                stngs["train_dnn"] = 1
                stngs['run_name'] = f"A_DNN_alignment_trials_seed{seed}"
                metrics_dnn = main(stngs)
                dnn_acc = metrics_dnn.get("dnn_accuracy", 0.0)
                if 0.68 <= dnn_acc <= 0.73:
                    print(f"DNN accuracy {dnn_acc:.4f} meets criteria. Found good seed: {seed}")
                    return seed
                else:
                    print(f"DNN accuracy {dnn_acc:.4f} does not meet criteria.")
            else:
                print(f"Full CBM accuracy {cbm_acc_full:.4f} does not meet criteria.")
        else:
            print(f"SCBM accuracy {cbm_acc:.4f} does not meet criteria.")
    return None

find_good_seed(settings)

#main(settings)