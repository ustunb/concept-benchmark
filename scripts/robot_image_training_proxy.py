import os, json, time, pickle, getpass, pathlib
from pathlib import Path
import numpy as np
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

    params = {
        "samples_per_instance": int(S["samples_per_instance"]),
        "draw": bool(int(S["draw"])),
        "output_directory": S.get("image_dir", run_dir / "images"),
        "concepts": {
            "head_shape": ["square", "round"],
            "body_shape": ["square", "round"],             # proxy P1
            "has_knees": ["false", "true"],
            "has_elbows": ["false", "true"],
            "has_antennae": ["false", "true"],
            "ears_shape": ["square", "triangle"],          # proxy P2
            "mouth_type": ["closed", "open"],
            "hand_shape": [                                # NS2
                "round_circle", "round_oval", "round_oval2",
                "edgy_triangle", "edgy_square", "edgy_trapezoid",
            ],
            "foot_shape": [                                # NS1
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
        "proxy_spec": S.get("proxy_spec", {}),
    }

    data = create_synthetic_dataset(**params)
    data.transform = tf
    data.generate_cvindices(seed=int(S["seed"]))

    # Compute drop list depending on mode
    mode = str(S.get("mode", "real")).lower()
    if mode == "oracle":
        # keep only coarse NS1/NS2; drop proxies and subtypes
        drop_list = ["body_shape", "ears_shape",
                     "foot_shape_flat_trapezoid","foot_shape_flat_rounded","foot_shape_flat_square","foot_shape_flat_5sided","foot_shape_flat_lshaped",
                     "foot_shape_pointy_trapezoid","foot_shape_pointy_rounded","foot_shape_pointy_square","foot_shape_pointy_3sided","foot_shape_pointy_4sided",
                     "hand_shape_round_circle","hand_shape_round_oval","hand_shape_round_oval2",
                     "hand_shape_edgy_triangle","hand_shape_edgy_square","hand_shape_edgy_trapezoid"]
    else:
        # hide coarse NS1/NS2; keep proxies + subtypes
        drop_list = ["foot_shape","hand_shape"]

    if S.get("skew_concept"):
        train, valid, test = create_skewed_splits(
            data,
            skew_specs=S["skew_concept"],
            rng=rng,
            drop_concepts=drop_list,
            fractions_unique=True
        )
    elif S.get("dataset_characterization", "") != "":
            train, valid, test = filter_training_by_string(data, string=S["dataset_characterization"], rng=rng)
    else:
            data.drop_concepts(drop_list)
            data.split("K05N01", fold_num_validation=4, fold_num_test=5)
            train = data.training; valid = data.validation; test = data.test
    
    ids = train.meta.get("robot_ids")
    if ids is not None:
            keep = np.unique(ids, return_index=True)[1]
            m = np.zeros(len(train.C), dtype=bool)
            m[keep] = True
            train = train.filter(m)

    # Basic stats
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

    # Device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    config = {
        'device': device,
        'batch_size': 32,
        'num_workers': 0 if device == 'mps' else 12,
        'pin_memory': False if device == 'mps' else True,
    }

    # Concept detector
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
        cd.fit(train, valid, embed_params={'shuffle': False, **config}, fit_params={"epochs": 50, 'lr': 1e-3, "patience": 10, **config})
        det_path = Path(settings["out_dir"]) / (S["run_name"] or "run") / det_name
        torch.save(cd.state_dict(), det_path)

    # Optional FE harness
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

    # Predict concepts
    P_tr = cd.predict(train, embed_params={"device": device})
    P_vl = cd.predict(valid, embed_params={"device": device})
    P_te = cd.predict(test,  embed_params={"device": device})
    H_tr = (P_tr > 0.5).astype(np.float32)
    H_te = (P_te > 0.5).astype(np.float32)
    H_vl = (P_vl > 0.5).astype(np.float32)

    # Front-end
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

    print("=== Learned Frontend Weights ===")
    for i, concept in enumerate(test.concepts):
        print(f"  {concept}: {fe.model.coef_[0, i]:.4f}")
    print(f"  bias: {fe.model.intercept_[0]:.4f}")

    # DNN baseline (optional)
    dnn_stats = {}
    if S.get("train_dnn", 0):
        print("Training baseline DNN...")
        paths_tr = [train.base_dir / p for p in train.X]; ytr = train.y.astype(int)
        paths_te = [test.base_dir / p for p in test.X];   yte = test.y.astype(int)
        dnn_acc, proc, dnn_model = train_eval_image(paths_tr, ytr, paths_te, yte,
                                                    model_id=S.get("image_model", "google/vit-base-patch16-224"),
                                                    epochs=int(S["epochs"]), batch_size=16, lr=5e-5, device=device)
        dnn_stats = {"dnn_accuracy": float(dnn_acc)}
        print(f"DNN accuracy: {float(dnn_acc)}")
        dnn_name = f"dnn_proxy_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"
        dnn_path = Path(settings["out_dir"]) / (S["run_name"] or "run") / dnn_name
        torch.save({"model_state_dict": dnn_model.state_dict(), "processor": proc}, dnn_path)

    # Interventions
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
    args, _ = parser.parse_known_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    for key in ['drop_concepts','skew_concept','proxy_spec']:
        if key in overrides:
            overrides[key] = json.loads(overrides[key])
    settings.update(overrides)
    main(settings)
