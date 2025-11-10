import json, time, pickle
from pathlib import Path
import numpy as np
import torch
import copy
import optuna
from itertools import combinations, product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from pathlib import Path
from torchvision import transforms
from concept_benchmark.models import ConceptDetector, FrontEndModel, RobotConceptClassifier, FrontEndModelCVXPY
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot import create_synthetic_dataset
from concept_benchmark.models import ConceptBasedModel
from concept_benchmark.intervention import (
    InterventionConfig,
    ConceptInterventionRunner,
)
from concept_benchmark.kflip import KFlipInterventionStrategy as ScoreIntervention
from scripts.dataset_skewing import create_skewed_splits, filter_training_by_string
from scripts.dnn_training import train_eval_image
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
    "train_dnn": 1,
    "seed": 1012,
    "model": "'glorp' if (int(row['mouth_type']=='closed') + int(row['foot_shape']=='pointy') + int(row['has_knees']=='true'))>= 3 else 'drent'",
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
        "signs": {
            "has_knees": 1
        },
        # "features": [
        #     (["foot_shape_flat_square"], ["True"], ">=", 0.95),
        #     (["foot_shape_flat_trapezoid"], ["True"], ">=", 0.95),
        #     #(["foot_shape_flat_square", "mouth_type"], ["True", "closed"], ">=", 0.95),
        #     #(["foot_shape_pointy_rounded", "mouth_type"], ["True", "open"], "<=", 0.05),
        #     #(["foot_shape_pointy_square", "mouth_type"], ["True", "open"], "<=", 0.05)
        # ]
    },
    "model_type": "stochastic",
    "logit_scalar": 4.2,
    "logit_intercept": -2,
    "logit_weights": {"mouth_type": 5, "foot_shape": 8, "has_knees": -5},
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
    "budget": [3],
    "intervention_accuracy": 1.0,
    "intervention_threshold": 0.1,
    "epochs": 1,
    "out_dir": str(results_dir / "robots"),
    "run_name": "scbm_run_1002_newer_alignment_trials",
    "load_detector": "",#str(Path(results_dir / "robots" / "cbm_run_1002_newer_alignment" / "detector_dnn_robots_image_stochastic_complete__skewint-acc100_seed1005.pt")),
    "load_frontend": ""#,str(Path(results_dir / "robots" / "cbm_run_1002_subconcepts" / "frontend_logreg_robots_image_stochastic_complete__skewint-acc90_seed1002.pkl")),
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
        test = concept_dataset.test


    if settings.get("label_noise_rate", 0.0) > 0:
        train = _apply_label_noise(train, settings["label_noise_rate"], seed=int(settings["seed"]))
        valid = _apply_label_noise(valid, settings["label_noise_rate"], seed=int(settings["seed"]))
        test = _apply_label_noise(test, settings["label_noise_rate"], seed=int(settings["seed"]))

    if missingness != "complete" and rate > 0:
        Ctr = _apply_missing(train.C, missingness, rate, rng, y=train.y.astype(int))
        train = train.__class__(parent=train.parent, X=train.X, C=Ctr, y=train.y, meta=train.meta,
                                transform=train.transform, concept_transform=train.concept_transform,
                                target_transform=train.target_transform, base_dir=train.base_dir)


    #print distribution of concepts in the test set:
    for c in test.concepts:
        unique, counts = np.unique(test.C[:, test.concepts.index(c)], return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"Concept '{c}' distribution in test set: {dist}")
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
                   seed_tag, skew_tag, test, train, monotonicity_constraints={}, prediction_constraints=[]):
    cvxpy = False
    if monotonicity_constraints or prediction_constraints:
        cvxpy = True
        fe = FrontEndModelCVXPY(monotonicity_constraints=monotonicity_constraints,
                                prediction_constraints=prediction_constraints,
                                concept_names=test.concepts,
                                concept_pos_value_map=test.meta['concept_pos_value'])
        fe_name = f"frontend_aligned_logreg_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pkl"
    else:
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

    if not cvxpy:
        print("\n=== Learned Frontend Weights ===")
        for i, concept in enumerate(test.concepts):
            print(f"  {concept}: {fe.model.coef_[0, i]:.4f}")
        print(f"  bias: {fe.model.intercept_[0]:.4f}")
    return acc_det, acc_gt, concept_acc_mean, fe, fe_path, y_pred_det


def test_alignment(fe, h_test, h_train, prob_train, prob_test, sttngs, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir,
                   seed_tag, skew_tag, test, train, monotonicity_constraints, prediction_constraints):
    test_concepts = h_test
    test_labels = test.y.astype(int)
    original_frontend = fe
    print(f"Aligning the model with the following constraints: {monotonicity_constraints} {prediction_constraints}")
    acc_det, _, _, aligned_frontend, _, _ = train_frontend(h_test, h_train, prob_train, sttngs, int_acc_tag, label_noise_tag,
                                                     miss_tag, model_type_tag, run_dir, seed_tag, skew_tag, test, train,
                                                     monotonicity_constraints, prediction_constraints)

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

    feweights = {}
    for i, concept in enumerate(test.concepts):
        feweights[concept] = round(aligned_frontend.model.coef_[0, i], 4)
    feweights["bias"] = round(aligned_frontend.model.intercept_[0], 4)

    aligned_preds = aligned_frontend.predict(h_test)

    # compute model accuracies per concept
    subtype_concepts = [c for c in test.concepts if c.startswith('foot_shape_')]
    missing_concepts = [c for c in sttngs.get("drop_concepts", []) if c.startswith('foot_shape_')]
    all_preds, confusion_df = _get_confusion_matrix(subtype_concepts, missing_concepts, aligned_frontend, h_test, prob_test, test)
    per_concept_acc = _get_accuracies_per_subconcept(all_preds, missing_concepts, subtype_concepts)

    # testing interventions
    _, _, intervention_results = test_interventions(prob_test, sttngs, acc_det, aligned_frontend, test)

    alignment_stats = {
        'original_accuracy': float(original_acc),
        'aligned_accuracy': float(aligned_acc),
        'accuracy_change': float(aligned_acc - original_acc),
        'predictions_changed': int(np.sum(original_preds != aligned_preds)),
        'frontend_weights': feweights,
        'model_accuracies_per_concept': per_concept_acc,
        'interventions': intervention_results
    }
    return alignment_stats


def train_dnn(sttngs, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir, seed_tag,
              skew_tag, test, train, tf):
    print("Training baseline DNN...")

    # Convert ConceptDatasetSample to path arrays
    paths_tr = [train.base_dir / p for p in train.X]
    ytr = train.y.astype(int)
    paths_te = [test.base_dir / p for p in test.X]
    yte = test.y.astype(int)

    dnn_acc, dnn_model = train_eval_image(
        paths_tr, ytr, paths_te, yte,
        epochs=int(sttngs["epochs"]),
        batch_size=16,
        lr=5e-5,
        device=device,
        seed=int(sttngs["seed"]),
        tf=tf,
        input_size=600 if sttngs["image_size"] == "large" else 32 if sttngs["image_size"] == "medium" else 8
    )

    dnn_stats = {"dnn_accuracy": float(dnn_acc)}
    print(f"DNN accuracy: {float(dnn_acc)}")

    dnn_name = f"dnn_vitb16_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"
    dnn_path = run_dir / dnn_name
    torch.save({
        "model_state_dict": dnn_model.state_dict(),
    }, dnn_path)
    return dnn_stats


def test_interventions(prob_test, sttngs, acc_det, fe, test):
    """Test interventions using the intervention framework."""
    intervention_results = {}
    rng = np.random.default_rng(int(sttngs["seed"]))
    budgets = sttngs.get('budget', [1])
    human_acc = sttngs.get("intervention_accuracy", 0.9)
    err_prob = 1.0 - human_acc

    # Create a CBM wrapper for the intervention framework
    cbm = ConceptBasedModel(concept_detector=None, front_end_model=fe)
    runner = ConceptInterventionRunner(cbm)

    for budget in budgets:
        config = InterventionConfig(
            max_concepts_per_instance=budget,
            random_state=int(sttngs["seed"]),
            score_threshold=sttngs.get("intervention_threshold", 1.0),
            noise = 1.0 - human_acc,
        )

        strategy = ScoreIntervention()

        # Run intervention
        result = runner.run(
            strategy=strategy,
            config=config,
            dataset=test,
            concept_proba=prob_test,
            labels=test.y.astype(int)
        )

        mask = result.mask
        C_gt = test.C.astype(np.float32)
        C_after = result.C_intervened.copy()  # GT at masked entries

        mistake_draw = rng.random(C_after.shape) < err_prob
        mistakes = mask & mistake_draw
        C_after[mistakes] = 1.0 - C_gt[mistakes]
        result.C_intervened = C_after


        # Extract intervention statistics
        n_intervened = np.sum(result.mask)
        n_samples = prob_test.shape[0]

        intervened_concepts = np.any(result.mask, axis=0)

        C_pred_binary = (result.C_pred >= 0.5).astype(int)
        C_final_binary = (result.C_intervened >= 0.5).astype(int)
        actual_edits_mask = (C_pred_binary != C_final_binary)
        result.y_prob_after = fe.predict_proba(C_final_binary)
        result.y_pred_after = np.argmax(result.y_prob_after, axis=1)

        acc_intervened = float((result.y_pred_after == test.y.astype(int)).mean())

        prediction_num_concepts_intervened_on = {int(i): int(np.sum(actual_edits_mask[i])) for i in range(n_samples)}
        concept_intervention_counts = {
            c: f"{int(np.sum(result.mask[:, i]))} ({int(np.sum(actual_edits_mask[:, i]))})"
            for i, c in enumerate(test.concepts) if intervened_concepts[i]
        }

        key = f"top_{budget}_human_acc_{int(human_acc * 100)}"
        intervention_results[key] = {
            "accuracy": acc_intervened,
            "accuracy_gain": acc_intervened - acc_det,
            "predictions_intervened_on": int(np.sum(np.any(result.mask, axis=1))),
            "interventions_rate": float(np.sum(np.any(result.mask, axis=1)) / n_samples),
            "avg_edits_per_intervention": float(sum(prediction_num_concepts_intervened_on.values())) / n_samples,
            "total_concept_checks": int(n_intervened),
            "total_concept_edits_made": sum(prediction_num_concepts_intervened_on.values()),
            "concept_interventions": concept_intervention_counts,
            "human_accuracy": human_acc
        }

    return budgets, human_acc, intervention_results


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
        dnn_stats = train_dnn(S, device, int_acc_tag, label_noise_tag, miss_tag, model_type_tag, run_dir,
                              seed_tag, skew_tag, test, train, tf)
    ###########################################################################

    ###########################################################################
    ###########################    INTERVENTIONS    ###########################
    ###########################################################################
    # apply interventions and measure their effect on the predictions
    budgets, human_acc, intervention_results = test_interventions(P_te, S, acc_det, fe, test)

    subtype_concepts = [c for c in test.concepts if c.startswith('foot_shape_')]
    missing_concepts = [c for c in S.get("drop_concepts", []) if c.startswith('foot_shape_')]

    # get the confusion matrix for the detector
    all_preds, confusion_df = _get_confusion_matrix(subtype_concepts, missing_concepts, fe, H_te, P_te, test)

    # get model accuracy for each subtype concept:
    per_concept_acc2 = _get_accuracies_per_subconcept(all_preds, missing_concepts, subtype_concepts)

    # align the model with new weights
    alignment_stats = {}
    if S.get("human_alignment", {}) != None:
        sign_alignment = S["human_alignment"].get("signs", {})
        prediction_alignment = S["human_alignment"].get("features", [])
        alignment_stats = test_alignment(fe, H_te, H_tr, P_tr, P_te, S, int_acc_tag, label_noise_tag, miss_tag, model_type_tag,
                                         run_dir, seed_tag, skew_tag, test, train, sign_alignment, prediction_alignment)
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


# a function that tries out different combinations of concepts in the sign human alignment constraint and different sigsn
# positive / negative, and searches for the optimum alignment where the accuracy of the model after interventions before
# alignment and after alignemnt si the highest AND the alignment does not break the accurayc of the model by more than 4%


def alignment_hyperparam_search(base_settings, max_iterations=50, n_random_init=10):
    """
    Search for optimal sign constraint combinations that maximize intervention accuracy drop
    while keeping model accuracy drop below 4%.

    Args:
        base_settings: The base settings dictionary to modify
        max_iterations: Maximum number of optimization iterations
        n_random_init: Number of random initializations before GP optimization

    Returns:
        dict: Best configuration found and search results
    """

    feature_names = [
        "mouth_type",
        "has_knees",
        "head_shape",
        "body_shape",
        "has_antennae",
        "ears_shape"
    ]

    signs = [-1, 1]

    evaluations = []
    best_config = None
    best_score = -np.inf

    def encode_config(sign_constraints):
        """Encode sign constraints as feature vector for GP"""
        # Each feature can be: no constraint (0), negative (-1), or positive (+1)
        encoding = np.zeros(len(feature_names))
        for i, feature in enumerate(feature_names):
            if feature in sign_constraints:
                encoding[i] = sign_constraints[feature]
        return encoding

    def evaluate_config(sign_constraints):
        """Evaluate a single configuration"""
        settings = copy.deepcopy(base_settings)
        settings["human_alignment"]["signs"] = sign_constraints
        settings["run_name"] = f"alignment_search_{len(evaluations)}"

        results = main(settings)

        # Extract metrics
        original_acc = results["cbm_acc_detected"]
        aligned_acc = results["alignment"]["aligned_accuracy"]
        original_int_acc = results["interventions"]["top_3_human_acc_90"]["accuracy"]
        aligned_int_acc = results["alignment"]["interventions"]["top_3_human_acc_90"]["accuracy"]

        # Calculate drops
        model_acc_drop = original_acc - aligned_acc
        intervention_acc_drop = original_int_acc - aligned_int_acc

        # Objective: maximize intervention accuracy drop, penalize model accuracy drop
        if model_acc_drop > 0.04:  # Model accuracy drops more than 4%
            score = -1000  # Heavy penalty
        else:
            # Reward intervention accuracy drop, small penalty for model drop
            score = intervention_acc_drop - 0.1 * model_acc_drop

        evaluation = {
            "sign_constraints": sign_constraints,
            "original_acc": float(original_acc),
            "aligned_acc": float(aligned_acc),
            "original_int_acc": float(original_int_acc),
            "aligned_int_acc": float(aligned_int_acc),
            "model_acc_drop": float(model_acc_drop),
            "intervention_acc_drop": float(intervention_acc_drop),
            "score": float(score),
            "valid": bool(model_acc_drop <= 0.04)
        }

        print(f"Config {len(evaluations)}: {sign_constraints}")
        print(f"  Model acc: {original_acc:.3f} -> {aligned_acc:.3f} after alignment (drop: {model_acc_drop:.3f})")
        print(f"  Int acc: {original_int_acc:.3f} -> {aligned_int_acc:.3f} after alignment (drop: {intervention_acc_drop:.3f})")
        print(f"  Score: {score:.3f}, Valid: {evaluation['valid']}")

        return evaluation

    # Random initialization phase
    print("=== Random Initialization Phase ===")
    rng = np.random.default_rng()
    for i in range(n_random_init):
        # Randomly select 1-3 features to constrain
        n_constraints = rng.integers(1, 4)
        selected_features = rng.choice(feature_names, size=n_constraints, replace=False)
        sign_constraints = {str(feature): int(np.random.choice(signs)) for feature in selected_features}
        print(f"n_constraints: {n_constraints}, selected_features: {selected_features}")
        print(f"\nEvaluating random config {i}: {sign_constraints}\n\n\n\n")

        evaluation = evaluate_config(sign_constraints)
        evaluations.append(evaluation)

        if evaluation.get("valid", False) and evaluation["score"] > best_score:
            best_score = evaluation["score"]
            best_config = evaluation

    # Prepare data for GP optimization
    X_train = np.array([encode_config(eval["sign_constraints"]) for eval in evaluations])
    y_train = np.array([eval["score"] for eval in evaluations])

    # Initialize GP
    kernel = Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    # Bayesian optimization phase
    print(f"\n=== Bayesian Optimization Phase ===")
    for iteration in range(max_iterations - n_random_init):
        # Fit GP to current data
        gp.fit(X_train, y_train)

        # Generate candidate configurations
        candidates = []
        candidate_encodings = []

        # Systematic exploration of feature combinations
        for n_constraints in range(1, 5):  # 1-3 constraints
            for features in combinations(feature_names, n_constraints):
                for sign_combo in product(signs, repeat=n_constraints):
                    sign_constraints = dict(zip(features, sign_combo))
                    encoding = encode_config(sign_constraints)
                    candidates.append(sign_constraints)
                    candidate_encodings.append(encoding)

        candidate_encodings = np.array(candidate_encodings)

        # Predict mean and std for all candidates
        mu, sigma = gp.predict(candidate_encodings, return_std=True)

        # Acquisition function: Upper Confidence Bound
        acquisition = mu + 2.0 * sigma

        # Select best candidate not yet evaluated
        for best_idx in np.argsort(-acquisition):
            candidate_config = candidates[best_idx]

            # Check if already evaluated
            already_evaluated = any(
                eval["sign_constraints"] == candidate_config
                for eval in evaluations
            )

            if not already_evaluated:
                break
        else:
            print("All reasonable candidates have been evaluated")
            break

        # Evaluate selected candidate
        evaluation = evaluate_config(candidate_config)
        evaluations.append(evaluation)

        # Update training data
        X_train = np.vstack([X_train, encode_config(evaluation["sign_constraints"])])
        y_train = np.append(y_train, evaluation["score"])

        # Update best configuration
        if evaluation.get("valid", False) and evaluation["score"] > best_score:
            best_score = evaluation["score"]
            best_config = evaluation
            print(f"  *** NEW BEST CONFIG! Score: {best_score:.3f}")

    # Summary
    valid_evaluations = [e for e in evaluations if e.get("valid", False)]

    print(f"\n=== Search Complete ===")
    print(f"Total evaluations: {len(evaluations)}")
    print(f"Valid configurations: {len(valid_evaluations)}")

    if best_config:
        print(f"\nBest configuration:")
        print(f"  Sign constraints: {best_config['sign_constraints']}")
        print(f"  Model accuracy drop: {best_config['model_acc_drop']:.3f}")
        print(f"  Intervention accuracy drop: {best_config['intervention_acc_drop']:.3f}")
        print(f"  Score: {best_config['score']:.3f}")
    else:
        print("No valid configuration found!")

    # Save results
    results = {
        "best_config": best_config,
        "all_evaluations": evaluations,
        "search_params": {
            "max_iterations": max_iterations,
            "n_random_init": n_random_init,
            "feature_names": feature_names,
        }
    }

    return results

main(settings)