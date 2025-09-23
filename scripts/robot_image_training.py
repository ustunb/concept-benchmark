import argparse, json, time, pickle
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from transformers import ViTModel
from concept_benchmark.models import ConceptDetector, FrontEndModel
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot import create_synthetic_dataset

settings = {
    "samples_per_instance": 1,
    "draw": 0,
    "image_size": 224,
    "color_mode": "color",
    "seed": 42,
    "missingness": "complete",
    "missing_rate": 0.0,
    "impute_missing": 1,
    "epochs": 10,
    "out_dir": str(results_dir / "robots"),
    "run_name": "",
    "load_detector": "",
    "load_frontend": "",
}

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

class ViTWrapper(torch.nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.vit = model if model else ViTModel.from_pretrained("google/vit-base-patch16-224")
    def forward(self, x):
        o = self.vit(pixel_values=x)
        return o.last_hidden_state[:, 0, :]

def _rate_tag(r):
    v = int(round(float(r) * 100))
    return f"{v:03d}"

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--samples-per-instance", type=int)
    p.add_argument("--draw", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--color-mode", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--missingness", type=str)
    p.add_argument("--missing-rate", type=float)
    p.add_argument("--impute-missing", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--out-dir", type=str)
    p.add_argument("--run-name", type=str)
    p.add_argument("--load-detector", type=str)
    p.add_argument("--load-frontend", type=str)
    args, _ = p.parse_known_args()
    if args.samples_per_instance is not None: settings["samples_per_instance"] = args.samples_per_instance
    if args.draw is not None: settings["draw"] = args.draw
    if args.image_size is not None: settings["image_size"] = args.image_size
    if args.color_mode is not None: settings["color_mode"] = args.color_mode
    if args.seed is not None: settings["seed"] = args.seed
    if args.missingness is not None: settings["missingness"] = args.missingness
    if args.missing_rate is not None: settings["missing_rate"] = args.missing_rate
    if args.impute_missing is not None: settings["impute_missing"] = args.impute_missing
    if args.epochs is not None: settings["epochs"] = args.epochs
    if args.out_dir is not None: settings["out_dir"] = args.out_dir
    if args.run_name is not None: settings["run_name"] = args.run_name
    if args.load_detector is not None: settings["load_detector"] = args.load_detector
    if args.load_frontend is not None: settings["load_frontend"] = args.load_frontend

    S = dict(settings)
    rng = np.random.default_rng(int(S["seed"]))
    base_out = Path(S["out_dir"]); base_out.mkdir(parents=True, exist_ok=True)
    miss = str(S["missingness"]).lower()
    rate = float(S["missing_rate"])
    miss_tag = "complete" if miss == "complete" or rate <= 0 else f"{miss}{_rate_tag(rate)}"
    impute_tag = f"impute{int(S['impute_missing'])}"
    seed_tag = f"seed{int(S['seed'])}"
    slug = f"robots_image_{miss_tag}_{seed_tag}_{impute_tag}"
    if S["run_name"]:
        run_dir = base_out / S["run_name"]
    else:
        run_dir = base_out / f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    IMG_SIZE = int(S["image_size"])
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    params = {
        "samples_per_instance": int(S["samples_per_instance"]),
        "draw": bool(int(S["draw"])),
        "output_directory": run_dir,
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
                "flat_4sided",
                "flat_5sided",
                "flat_lshaped",
                "pointy_3sided",
                "pointy_4sided",
                "pointy_6sided",
            ],
        },
        "spurious_features": ["has_elbows", "hand_shape"],
        "model": "('glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy_')))>=1 else 'drent')",
        "model_type": "deterministic",
        "size": "large",
        "color_mode": str(S["color_mode"]),
        "train_concept_detector": False,
        "epochs": int(S["epochs"]),
        "verbose": True,
    }

    data = create_synthetic_dataset(**params)
    data.transform = tf
    data.generate_cvindices(seed=int(S["seed"]))
    data.split("K05N01", fold_num_validation=4, fold_num_test=5)
    train = data.training; valid = data.validation; test = data.test

    if miss != "complete" and rate > 0:
        Ctr = _apply_missing(train.C, miss, rate, rng, y=train.y.astype(int))
        train = train.__class__(parent=train.parent, X=train.X, C=Ctr, y=train.y, meta=train.meta, transform=train.transform, concept_transform=train.concept_transform, target_transform=train.target_transform)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    cd = ConceptDetector(embedding_model=ViTWrapper())
    det_name = f"detector_dnn_vitb16_robots_image_{miss_tag}_{seed_tag}.pt"
    if S["load_detector"]:
        state = torch.load(S["load_detector"], map_location="cpu")
        cd.load_state_dict(state)
        det_path = Path(S["load_detector"])
    else:
        cd.fit(train, valid, freeze=True, embed_params={"device": device}, fit_params={"epochs": int(S["epochs"]), "device": "cpu"})
        det_path = run_dir / det_name
        torch.save(cd.state_dict(), det_path)

    P_tr = cd.predict_proba(train, embed_params={"device": device})
    P_te = cd.predict_proba(test, embed_params={"device": device})
    H_tr = (P_tr > 0.5).astype(np.float32)
    H_te = (P_te > 0.5).astype(np.float32)

    fe = FrontEndModel()
    fe_name = f"frontend_logreg_robots_image_{miss_tag}_{seed_tag}.pkl"
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
            keep = np.all(Ctr >= 0, axis=1)
            fe.fit(H_tr[keep], train.y[keep].astype(int))
        fe_path = run_dir / fe_name
        with open(fe_path, "wb") as f:
            pickle.dump(fe, f)

    y_pred_det = fe.predict_proba(H_te)
    y_pred_gt = fe.predict_proba(test.C.astype(np.float32))
    acc_det = float((y_pred_det.argmax(1) == test.y.astype(int)).mean())
    acc_gt = float((y_pred_gt.argmax(1) == test.y.astype(int)).mean())
    concept_acc_mean = float((H_te == test.C).mean())

    meta_name = f"meta_cbm_detected_robots_image_{miss_tag}_{seed_tag}.json"
    metrics_name = f"metrics_cbm_detected_robots_image_{miss_tag}_{seed_tag}.json"

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
        "concepts": list(data.parent.concepts.keys()) if hasattr(data.parent, "concepts") else [],
        "naming_slug": slug,
    }

    metrics = {
        "cbm_acc_detected": acc_det,
        "cbm_acc_oracle": acc_gt,
        "concept_det_acc_mean": concept_acc_mean,
    }

    meta_path = run_dir / meta_name
    metrics_path = run_dir / metrics_name
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

main()
