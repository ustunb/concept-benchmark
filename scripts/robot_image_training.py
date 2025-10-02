import argparse, json, time, pickle
from pathlib import Path
import numpy as np
import torch
import copy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    ViTModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from concept_benchmark.models import ConceptDetector, FrontEndModel, RobotConceptClassifier
from concept_benchmark.paths import results_dir
from concept_benchmark.synthetic.robot import create_synthetic_dataset

settings = {
    "samples_per_instance": 1,
    "draw": 0,
    "CBM_type": "joint", #"sequential"
    "image_dir": "./data/robot_images",
    "image_size": "large",
    "color_mode": "greyscale",
    "train_dnn": 0,
    "seed": 42,
    "model": "'glorp' if (int(row['mouth_type']=='closed') + int(row['foot_shape']=='pointy') + int(row['body_shape']=='square'))>= 2 else 'drent'",
    'dataset_characterization': "",
    "knows_concepts": True,
    "human_alignment": {"foot_shape": -1, "body_shape": -1, "mouth_type": -1, "bias": 2}, # OR of ANDs model's logic
    "model_type": "deterministic",
    "label_noise_rate": 0.0,
    "missingness": "complete",
    "missing_rate": 1.0,
    "impute_missing": 0,
    "skew_concept": [{'concepts': {'body_shape': 0, 'foot_shape': 1, 'has_antennae': 1}, 'min_fraction': 0.3},
                     {'concepts': {'mouth_type': 0, 'foot_shape': 1, 'has_antennae': 1}, 'min_fraction': 0.3},
                     {'concepts': {'body_shape': 0, 'mouth_type': 0, 'has_antennae': 1}, 'min_fraction': 0.3},
                     {'concepts': {'body_shape': 1, 'mouth_type': 1, 'has_antennae': 0}, 'min_fraction': 0.3},
                     {'concepts': {'body_shape': 1, 'foot_shape': 0, 'has_antennae': 0}, 'min_fraction': 0.3},
                     {'concepts': {'foot_shape': 0, 'mouth_type': 1, 'has_antennae': 0}, 'min_fraction': 0.3}],#[{'concepts': {'mouth_type': 0, 'foot_shape_pointy_3sided': 1}, 'min_fraction': 0.13},
                    # {'concepts': {'mouth_type': 0, 'foot_shape_pointy_4sided': 1}, 'min_fraction': 0.13},
                    # {'concepts': {'mouth_type': 1, 'foot_shape_pointy_3sided': 1}, 'min_fraction': 0.13},
                    # {'concepts': {'mouth_type': 1, 'foot_shape_pointy_4sided': 1}, 'min_fraction': 0.13},
                    # {'concepts': {'mouth_type': 1, 'foot_shape_flat_lshaped': 1}, 'min_fraction': 0.13},
                    # {'concepts': {'mouth_type': 1, 'foot_shape_flat_4sided': 1}, 'min_fraction': 0.13},
                    # {'concepts': {'mouth_type': 0, 'foot_shape_flat_lshaped': 1}, 'min_fraction': 0.13},
                    # {'concepts': {'mouth_type': 0, 'foot_shape_flat_4sided': 1}, 'min_fraction': 0.13}
                    # ],
    "budget": [9],
    "intervention_accuracy": 0.9,
    "epochs": 10,
    "out_dir": str(results_dir / "robots"),
    "run_name": "bias_antenna_largeimage_vanillaCD2",
    "load_detector": "",#str(Path(results_dir / "robots" / "bias_antenna_largeimage" / "detector_dnn_vitb16_robots_image_deterministic_complete__skewint-acc90_seed42.pt")),
    "load_frontend": "",#str(Path(results_dir / "robots" / "bias_antenna_largeimage" / "frontend_logreg_robots_image_deterministic_complete__skewint-acc90_seed42.pkl")),
}

class ImageDS(Dataset):
    def __init__(self, X_paths, y, proc):
        self.X = [str(p) for p in X_paths]
        self.y = np.asarray(y, dtype=int)
        self.proc = proc
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        img = Image.open(self.X[i]).convert("RGB")
        enc = self.proc(images=img, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(self.y[i], dtype=torch.long)
        return enc, y

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


def train_eval_image(paths_tr, y_tr, paths_te, y_te, model_id, epochs, batch_size, lr, device):
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id, num_labels=2, ignore_mismatched_sizes=True)
    ds_tr = ImageDS(paths_tr, y_tr, proc)
    ds_te = ImageDS(paths_te, y_te, proc)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in dl_tr:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb, labels=yb)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)
            out = model(**xb)
            pred = out.logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    acc = correct / total if total > 0 else 0.0
    return float(acc), proc, model

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


def create_sample(size, indices, dataset):
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return dataset._full.filter(mask)


def create_skewed_splits(dataset, skew_specs, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25, rng=None):
    """
    Skew training by ensuring minimum representation of specific concept patterns.

    Args:
        skew_specs: List of dicts, each with 'concepts' (dict of concept:value) and 'min_fraction' (float)
                   e.g., [{'concepts': {'body_shape': 0, 'foot_shape_3sided': 1}, 'min_fraction': 0.4},
                          {'concepts': {'body_shape': 0, 'foot_shape_4sided': 1}, 'min_fraction': 0.4}]
    """
    if rng is None:
        rng = np.random.default_rng()

    # print y labels all
    print("Overall class distribution in full dataset:")
    unique, counts = np.unique(dataset.y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    for cls, cnt in class_dist.items():
        print(f"  Class {cls}: {cnt} samples ({cnt / len(dataset.y):.1%})")

    total_size = len(dataset.C)
    desired_train_size = int(total_size * train_fraction)
    print("Desired training size:", desired_train_size)

    # Find indices matching each specification
    spec_indices = []
    for spec in skew_specs:
        mask = np.ones(total_size, dtype=bool)
        for concept_name, target_value in spec['concepts'].items():
            concept_idx = dataset.concepts.index(concept_name)
            mask &= (dataset.C[:, concept_idx] == target_value)
        spec_indices.append(np.where(mask)[0])

    train_indices = []
    used = set()
    for spec, indices in zip(skew_specs, spec_indices):
        needed = int(desired_train_size * spec['min_fraction'])
        available = [i for i in indices if i not in used]
        rng.shuffle(available)
        take = available[:min(needed, len(available))]
        train_indices.extend(take)
        used.update(take)
        print(f"Added {len(take)} for spec {spec['concepts']} (wanted {needed})")

    # Fill remaining slots with any unused samples
    remaining_slots = desired_train_size - len(train_indices)
    if remaining_slots > 0:
        unused = [i for i in range(total_size) if i not in used]
        rng.shuffle(unused)
        train_indices.extend(unused[:remaining_slots])
        print(f"Filled {min(remaining_slots, len(unused))} remaining slots")

    print("\n=== Debugging: Sample robots from each spec ===")
    for spec, indices in zip(skew_specs, spec_indices):
        print(f"\nSpec {spec['concepts']}: {len(indices)} total samples")
        print("Sample of 10 robots:")
        sample_indices = indices[:10] if len(indices) >= 10 else indices

        for sample_idx in sample_indices:
            robot_features = {}
            for i, concept_name in enumerate(dataset.concepts):
                robot_features[concept_name] = int(dataset.C[sample_idx, i])
                robot_features["class"] = int(dataset.y[sample_idx])
            print(f"  Robot {sample_idx}: {robot_features}")


    train_indices = np.array(train_indices)
    rng.shuffle(train_indices)

    # Validation and test from what's left
    remaining = np.array([i for i in range(total_size) if i not in train_indices])
    rng.shuffle(remaining)

    val_size = int(len(remaining) * val_fraction / (val_fraction + test_fraction))
    val_indices = remaining[:val_size]
    test_indices = remaining[val_size:]

    dataset.training = create_sample(total_size, train_indices, dataset)
    dataset.validation = create_sample(total_size, val_indices, dataset)
    dataset.test = create_sample(total_size, test_indices, dataset)

    return dataset.training, dataset.validation, dataset.test


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


def filter_training_by_string(dataset, string, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, rng=None):
    """
    Filter robots for training set based on model string, put rest in val/test.

    Args:
        dataset: ConceptDataset instance
        string: String condition to evaluate for training selection
        train_fraction, val_fraction, test_fraction: Split proportions
        rng: Random number generator

    Returns:
        train, validation, test splits
    """
    if rng is None:
        rng = np.random.default_rng()

    def create_row_dict(sample_idx):
        row = {}
        for i, concept_name in enumerate(dataset.concepts):
            concept_value = dataset.C[sample_idx, i]

            if concept_name == 'body_shape':
                row[concept_name] = 'square' if concept_value == 0 else 'round'
            elif concept_name == 'head_shape':
                row[concept_name] = 'square' if concept_value == 0 else 'round'
            elif concept_name in ['has_knees', 'has_elbows', 'has_antennae']:
                row[concept_name] = 'true' if concept_value == 1 else 'false'
            elif concept_name == 'ears_shape':
                row[concept_name] = 'square' if concept_value == 0 else 'triangle'
            elif concept_name == 'mouth_type':
                row[concept_name] = 'closed' if concept_value == 0 else 'open'
            elif concept_name == 'hand_shape':
                row[concept_name] = 'round_circle' if concept_value == 0 else 'edgy_triangle'
            elif concept_name == 'foot_shape':
                row[concept_name] = 'flat_4sided' if concept_value == 0 else 'pointy_3sided'
            else:
                row[concept_name] = concept_value

        return row

    train_candidates = []
    other_samples = []

    for idx in range(len(dataset.C)):
        row = create_row_dict(idx)
        try:
            print(row)
            if eval(string, {"row": row}):
                print("  -> Train candidate")
                train_candidates.append(idx)
            else:
                other_samples.append(idx)
        except Exception as e:
            print(f"Error evaluating condition for sample {idx}: {e}")
            other_samples.append(idx)

    train_candidates = np.array(train_candidates)
    other_samples = np.array(other_samples)

    print(f"Candidates for training (satisfy condition): {len(train_candidates)}")
    print(f"Other samples: {len(other_samples)}")

    total_size = len(dataset.C)
    desired_train_size = int(total_size * train_fraction)

    actual_train_size = min(len(train_candidates), desired_train_size)
    rng.shuffle(train_candidates)
    train_indices = train_candidates[:actual_train_size]

    unused_candidates = train_candidates[actual_train_size:]
    remaining_samples = np.concatenate([unused_candidates, other_samples])
    rng.shuffle(remaining_samples)

    remaining_size = len(remaining_samples)
    val_size = int(remaining_size * val_fraction / (val_fraction + test_fraction))

    val_indices = remaining_samples[:val_size]
    test_indices = remaining_samples[val_size:]

    dataset.training = create_sample(total_size, train_indices, dataset)
    dataset.validation = create_sample(total_size, val_indices, dataset)
    dataset.test = create_sample(total_size, test_indices, dataset)

    print(f"\nFinal splits:")
    print(f"Training: {len(train_indices)} samples ({len(train_indices) / total_size:.1%})")
    print(f"Validation: {len(val_indices)} samples ({len(val_indices) / total_size:.1%})")
    print(f"Test: {len(test_indices)} samples ({len(test_indices) / total_size:.1%})")

    return dataset.training, dataset.validation, dataset.test


class ViTWrapper(torch.nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.vit = model if model else ViTModel.from_pretrained("google/vit-base-patch16-224")
    def forward(self, x):
        o = self.vit(pixel_values=x)
        return o.last_hidden_state[:, 0, :]


class CNNWrapper(torch.nn.Module):
    def __init__(self, cnn_model):
        super().__init__()
        self.cnn = cnn_model

    def forward(self, x):
        return self.cnn.backbone(x).flatten(1)


def _rate_tag(r):
    v = int(round(float(r) * 100))
    return f"{v:03d}"


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--samples-per-instance", type=int)
    p.add_argument("--draw", type=int)
    p.add_argument("--image-dir", type=str, help="Directory to save robot images")
    p.add_argument("--image-size", type=int)
    p.add_argument("--color-mode", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--missingness", type=str)
    p.add_argument("--missing-rate", type=float)
    p.add_argument("--skew-concept", type=str, nargs='+', help="Concept(s) to skew")
    p.add_argument("--skew-value", type=int, nargs='+', help="Value(s) to keep (0 or 1)")
    p.add_argument("--skew-fraction", type=float, nargs='+', help="Fraction(s) of data to keep")
    p.add_argument("--impute-missing", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--out-dir", type=str)
    p.add_argument("--run-name", type=str)
    p.add_argument("--load-detector", type=str)
    p.add_argument("--load-frontend", type=str)
    args, _ = p.parse_known_args()
    if args.samples_per_instance is not None: settings["samples_per_instance"] = args.samples_per_instance
    if args.draw is not None: settings["draw"] = args.draw
    if args.image_dir is not None: settings["image_dir"] = args.image_dir
    if args.image_size is not None: settings["image_size"] = args.image_size
    if args.color_mode is not None: settings["color_mode"] = args.color_mode
    if args.seed is not None: settings["seed"] = args.seed
    if args.missingness is not None: settings["missingness"] = args.missingness
    if args.missing_rate is not None: settings["missing_rate"] = args.missing_rate
    if args.impute_missing is not None: settings["impute_missing"] = args.impute_missing
    if args.skew_concept is not None: settings["skew_concept"] = args.skew_concept
    if args.skew_value is not None: settings["skew_value"] = args.skew_value
    if args.skew_fraction is not None: settings["skew_fraction"] = args.skew_fraction
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
    int_acc_tag = f"int-acc{int(round(float(S["intervention_accuracy"]) * 100))}"
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

    IMG_SIZE = 224
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                "flat_4sided",
                "flat_5sided",
                "flat_lshaped",
                "pointy_3sided",
                "pointy_4sided",
                "pointy_6sided",
            ],
        },
        "additional_features": [] if S.get("knows_concepts", True) else ["foot_shape_subtype"],
        "spurious_features": ["has_elbows", "hand_shape"],
        "model": S.get("model", "'glorp' if (int(row['body_shape']=='square') + int(str(row['foot_shape']).startswith('pointy')))>=1 else 'drent'"),
        "model_type": S["model_type"],
        "size": S["image_size"],
        "color_mode": str(S["color_mode"]),
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
            rng=rng
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

    if S.get("label_noise_rate", 0.0) > 0:
        train = _apply_label_noise(train, S["label_noise_rate"], seed=int(S["seed"]))
        valid = _apply_label_noise(valid, S["label_noise_rate"], seed=int(S["seed"]))
        test = _apply_label_noise(test, S["label_noise_rate"], seed=int(S["seed"]))

    if miss != "complete" and rate > 0:
        Ctr = _apply_missing(train.C, miss, rate, rng, y=train.y.astype(int))
        train = train.__class__(parent=train.parent, X=train.X, C=Ctr, y=train.y, meta=train.meta, transform=train.transform, concept_transform=train.concept_transform, target_transform=train.target_transform, base_dir=train.base_dir)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    cd = ConceptDetector(embedding_model=CNNWrapper(RobotConceptClassifier(num_concepts=train.n_concepts)))
    det_name = f"detector_dnn_robots_image_{model_type_tag}{miss_tag}{label_noise_tag}{skew_tag}{int_acc_tag}_{seed_tag}.pt"
    if S["load_detector"]:
        mini_train = train.filter(np.array([True] + [False] * (len(train.C) - 1)))
        mini_valid = valid.filter(np.array([True] + [False] * (len(valid.C) - 1)))

        cd.fit(mini_train, mini_valid, freeze=True, embed_params={"device": device},
               fit_params={"epochs": 1, "device": "cpu"})
        state = torch.load(S["load_detector"], map_location="cpu")
        cd.concept_layers.load_state_dict(state)
        det_path = Path(S["load_detector"])
    else:
        cd.fit(train, valid, freeze=True, embed_params={"device": device}, fit_params={"epochs": 50, "patience": 10,
                                                                                       "device": device})
        det_path = run_dir / det_name
        torch.save(cd.concept_layers.state_dict(), det_path)

    P_tr = cd.predict(train, embed_params={"device": device})
    P_te = cd.predict(test, embed_params={"device": device})
    H_tr = (P_tr > 0.5).astype(np.float32)
    H_te = (P_te > 0.5).astype(np.float32)

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

    # Add this after computing frontend predictions

    print("\n=== Analyzing test samples with GT concepts ===")

    # Get frontend scores (before sigmoid)
    def get_raw_scores(frontend, concepts):
        """Get raw logistic regression scores before sigmoid"""
        return frontend.model.decision_function(concepts)

    gt_scores = get_raw_scores(fe, test.C.astype(np.float32))
    det_scores = get_raw_scores(fe, H_te)
    gt_preds = fe.predict(test.C)

    # Create a dataframe for analysis
    import pandas as pd

    analysis_data = []
    for i in range(len(test.y)):
        row_data = {
            'sample_idx': i,
            'true_label': int(test.y[i]),
            'gt_pred': int(gt_preds[i]),
            'gt_score': float(gt_scores[i]),
            "det_score": float(det_scores[i]),
            'correct': int(gt_preds[i] == test.y[i]),
        }

        # Add all concept values
        for j, concept_name in enumerate(test.concepts):
            row_data[concept_name] = int(test.C[i, j])

        analysis_data.append(row_data)

    df_analysis = pd.DataFrame(analysis_data)

    # Sort by score to see what's happening
    df_sorted = df_analysis.sort_values('gt_score', ascending=False)

    print("\n=== Top 20 samples by GT score (should predict glorp) ===")
    print(df_sorted[['sample_idx', 'true_label', 'gt_pred', 'gt_score', 'correct',
                     'mouth_type', 'foot_shape', 'body_shape', 'has_antennae']].head(
        20).to_string(index=False))

    print("\n=== Bottom 20 samples by GT score (should predict drent) ===")
    print(df_sorted[['sample_idx', 'true_label', 'gt_pred', 'gt_score', 'det_score', 'correct',
                     'mouth_type', 'foot_shape', 'body_shape', 'has_antennae']].tail(
        20).to_string(index=False))

    print("\n=== Glorp samples (true_label=1) that were predicted as drent ===")
    wrong_glorps = df_analysis[(df_analysis['true_label'] == 1) & (df_analysis['gt_pred'] == 0)]
    print(f"Total: {len(wrong_glorps)}")
    print(wrong_glorps[['sample_idx', 'gt_score', 'det_score', 'mouth_type', 'foot_shape', 'body_shape', 'has_antennae']].head(30).to_string(index=False))

    # Check the decision boundary
    print(f"\n=== Decision boundary analysis ===")
    print(f"Min score for correct glorp prediction: {df_analysis[df_analysis['gt_pred'] == 1]['gt_score'].min():.4f}")
    print(f"Max score for correct drent prediction: {df_analysis[df_analysis['gt_pred'] == 0]['gt_score'].max():.4f}")
    print(
        f"Score distribution for true glorps: mean={df_analysis[df_analysis['true_label'] == 1]['gt_score'].mean():.4f}, median={df_analysis[df_analysis['true_label'] == 1]['gt_score'].median():.4f}")
    print(
        f"Score distribution for true drents: mean={df_analysis[df_analysis['true_label'] == 0]['gt_score'].mean():.4f}, median={df_analysis[df_analysis['true_label'] == 0]['gt_score'].median():.4f}")

    print("\n=== Learned Frontend Weights ===")
    for i, concept in enumerate(test.concepts):
        print(f"  {concept}: {fe.model.coef_[0, i]:.4f}")
    print(f"  bias: {fe.model.intercept_[0]:.4f}")

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
    def select_intervention_concepts(current_pred, frontend_model, budget_k):
        """Select concepts to intervene on based on prediction change probability."""
        n_concepts = len(current_pred)
        current_proba = frontend_model.predict_proba(current_pred.reshape(1, -1))[0]
        current_class = np.argmax(current_proba)

        concept_scores = []

        for j in range(n_concepts):
            # try flipping concept j to opposite value
            test_pred = current_pred.copy()
            test_pred[j] = 1 - current_pred[j]
            test_proba = frontend_model.predict_proba(test_pred.reshape(1, -1))[0]
            new_class = np.argmax(test_proba)

            # score = probability this intervention changes the prediction
            flip_probability = 1.0 if new_class != current_class else 0.0
            concept_scores.append((j, flip_probability))

        # sort by highest flip probability
        concept_scores.sort(key=lambda x: x[1], reverse=True)

        # select top budget_k concepts
        selected = [idx for idx, score in concept_scores[:budget_k]]
        return selected

    def apply_interventions(predictions, ground_truth, frontend_model, budget_k, human_accuracy=1.0,
                            policy="probability", rng=None):
        """
        Apply human interventions to concept predictions.

        Args:
            predictions: (n_samples, n_concepts) - Current concept predictions
            ground_truth: (n_samples, n_concepts) - True concept values
            frontend_model: Trained frontend model for final predictions
            budget_k: int - Max concepts to intervene on per sample
            human_accuracy: float - Probability human gives correct intervention
            policy: str - "uncertainty" or "oracle" intervention selection
            rng: np.random.Generator - For reproducibility

        Returns:
            intervened_predictions: (n_samples, n_concepts) - After interventions
            intervention_stats: dict - Statistics about interventions applied
        """
        if rng is None:
            rng = np.random.default_rng()

        H_intervened = predictions.copy().astype(float)
        n_samples, n_concepts = H_intervened.shape
        edit_counts = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            if budget_k <= 0:
                continue

            current_pred = H_intervened[i].copy()
            if policy == "oracle":
                # oracle: select concepts that are currently wrong
                wrong_concepts = [j for j in range(n_concepts)
                                  if abs(current_pred[j] - ground_truth[i, j]) > 0.5]
                if not wrong_concepts:
                    continue
                selected_concepts = wrong_concepts[:budget_k]
            else:
                # flip probability policy
                selected_concepts = select_intervention_concepts(current_pred, frontend_model, budget_k)

            actual_edits = 0
            for j in selected_concepts:
                original_value = current_pred[j]
                if rng.random() < human_accuracy:
                    # human gives correct value
                    new_value = ground_truth[i, j]
                else:
                    # human gives incorrect value
                    new_value = 1 - ground_truth[i, j]

                current_pred[j] = new_value

                # only count as edit if value actually changed
                if abs(new_value - original_value) > 1e-6:
                    actual_edits += 1

            H_intervened[i] = current_pred
            edit_counts[i] = actual_edits

        n_interventions = np.sum(edit_counts > 0)
        stats = {
            "predictions_intervened_on": int(n_interventions),
            "interventions_rate": float(n_interventions) / n_samples,
            "avg_edits_per_intervention": float(edit_counts[edit_counts > 0].mean()) if n_interventions > 0 else 0.0,
            "total_concept_checks": int(budget_k * n_samples),
            "total_concept_edits_made": int(edit_counts.sum())
        }

        return H_intervened, stats

    intervention_results = {}
    budgets = S.get('budget', [1, 2, 3, 4, 5])
    human_acc = S.get("intervention_accuracy", 1.0)
    for budget in budgets:
        for policy in ["oracle", "flip_probability"]:
            H_intervened, intervention_stats = apply_interventions(
                predictions=H_te,
                ground_truth=test.C.astype(np.float32),
                frontend_model=fe,
                budget_k=budget,
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
                "predictions_intervened_on": intervention_stats["predictions_intervened_on"],
                "interventions_rate": intervention_stats["interventions_rate"],
                "avg_edits_per_intervention": intervention_stats["avg_edits_per_intervention"],
                "total_concept_checks": intervention_stats["total_concept_checks"],
                "total_concept_edits_made": intervention_stats["total_concept_edits_made"],
                "policy": policy,
                "budget": budget,
                "human_accuracy": human_acc
            }

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
    metrics.update({'concept_accuracies': per_concept_acc, 'train_concept_accuracies': train_per_concept_acc})

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

