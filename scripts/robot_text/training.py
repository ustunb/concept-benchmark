"""Robot text pipeline — model training functions."""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from concept_benchmark.utils import determine_device, set_deterministic_seed
from concept_benchmark.config import RobotBenchmarkConfig
from concept_benchmark.data import ConceptDatasetSample
from concept_benchmark.ext.fileutils import load, save
from concept_benchmark.paths import package_dir
from experiments.models import ConceptBasedModel, FrontEndModel
from experiments.text_concept_detector import TextConceptDetector

logger = logging.getLogger(__name__)


def _internal_output_mode(concept_output_type: str) -> str:
    """Translate public config value to internal detector/LFCBM mode string."""
    _mapping = {"binary": "hard", "continuous": "soft"}
    if concept_output_type not in _mapping:
        raise ValueError(
            f"Unknown concept_output_type {concept_output_type!r}. "
            f"Valid values: {sorted(_mapping)}"
        )
    return _mapping[concept_output_type]


class _TextDS(Dataset):
    """Simple text dataset for DistilBERT fine-tuning."""

    def __init__(self, X, y, tok, max_length=256):
        self.X = list(map(str, X))
        self.y = np.asarray(y, dtype=int)
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        enc = self.tok(
            self.X[i],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(self.y[i], dtype=torch.long)
        return enc, y


def _train_dnn_text(X_tr, y_tr, X_te, y_te, model_id, epochs, batch_size, lr, device):
    """Train a text classifier (DistilBERT) and return accuracy + model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    ds_tr = _TextDS(X_tr, y_tr, tok)
    ds_te = _TextDS(X_te, y_te, tok)
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
            optim.zero_grad()
            out.loss.backward()
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
    return float(acc), tok, model


def _fit_platt(X, y, tok, model, device):
    """Fit Platt calibration on validation set."""
    from sklearn.linear_model import LogisticRegression

    ds = _TextDS(X, np.asarray(y, dtype=int), tok)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    z_list, y_list = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            out = model(**xb)
            z = (out.logits[:, 1] - out.logits[:, 0]).detach().cpu().numpy()
            z_list.append(z)
            y_list.append(yb.numpy())
    Z = np.concatenate(z_list).reshape(-1, 1) if z_list else np.zeros((0, 1))
    Y = np.concatenate(y_list).astype(int) if y_list else np.zeros(0, dtype=int)
    if Y.size == 0 or np.unique(Y).size < 2:
        return None
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(Z, Y)
    return lr


# ── Stage: setup_dataset ─────────────────────────────────────────────


def setup_dataset(
    config: RobotBenchmarkConfig,
) -> ConceptDatasetSample:
    """Generate robot text dataset and split by robot identity.

    Returns the dataset with training/validation/test splits set.
    """
    from concept_benchmark.generators import DatasetGenerator

    ds = DatasetGenerator.from_config(config).generate()

    from concept_benchmark.config import TEXT_PRESET_EXCLUDED_CONCEPTS
    excluded = TEXT_PRESET_EXCLUDED_CONCEPTS[config.concept_preset]
    if excluded:
        ds.drop_concepts(excluded)

    # Fractional split with group constraint: prevents the same robot identity
    # from appearing in both train and test (avoids data leakage).
    ds.sample(
        test_size=0.15,
        val_size=0.2,
        groups=ds.meta["row_index"],
        stratify=ds.y,
        seed=config.seed + 1,
    )

    # Missingness can be applied here if needed:
    # ds.sample_concept_missingness(p=0.2, mechanism="mcar", rng=config.seed + 999,
    #                                splits={"train"}, enable=True)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        ds.train.n,
        ds.validation.n,
        ds.test.n,
    )
    save(ds, config.get_dataset_path(), overwrite=True)
    return ds


# ── Stage: train_cbm ─────────────────────────────────────────────────


def train_cbm(
    config: RobotBenchmarkConfig,
    data: ConceptDatasetSample | None = None,
) -> ConceptBasedModel:
    """Train TextConceptDetector + FrontEndModel.

    Returns the trained ConceptBasedModel.
    """
    set_deterministic_seed(config.seed)

    if data is None:
        data = load(config.get_dataset_path())

    detector = TextConceptDetector(
        embed_dim=128,
        hidden_dim=192,
        epochs=config.detector_epochs,
        batch_size=config.detector_batch_size,
        lr=config.detector_lr,
        use_bigrams=True,
        dropout=0.1,
        pos_weight="auto",
        output_mode=_internal_output_mode(config.concept_output_type),
        threshold_mode="auto",
        pooling="attn",
        group_unknown_threshold=0.50,
        validate=True,
    )

    detector.fit(data.training, data.validation)

    cbm = ConceptBasedModel(
        concept_detector=detector,
        label_predictor=FrontEndModel(),
        should_propagate=(config.concept_output_type == "continuous"),
    )

    # Train frontend on ground-truth concepts
    cbm.label_predictor.fit(data.training.C, data.training.y)

    test_pred = cbm.predict(data.test)
    acc = float(np.mean(test_pred == data.test.y))
    logger.info("CBM Test Accuracy: %.4f", acc)

    save(cbm, config.get_model_path("cbm"), overwrite=True)
    return cbm


def train_cbm_subjective(
    config: RobotBenchmarkConfig,
    data: ConceptDatasetSample | None = None,
) -> ConceptBasedModel:
    """Train a CBM on noisy (subjective) concept labels.

    Saves to ``config.get_model_path("cbm_subjective")``.
    """
    import copy as _copy

    if data is None:
        data = load(config.get_dataset_path())
    noisy_data = _copy.deepcopy(data)
    noisy_data.sample_concept_noise(
        p=config.subjective_noise_rate,
        # Offset seed so noise RNG is independent of data-generation RNG.
        rng=np.random.default_rng(config.seed + 555),
        enable=True,
    )

    set_deterministic_seed(config.seed)
    detector = TextConceptDetector(
        embed_dim=128,
        hidden_dim=192,
        epochs=config.detector_epochs,
        batch_size=config.detector_batch_size,
        lr=config.detector_lr,
        use_bigrams=True,
        dropout=0.1,
        pos_weight="auto",
        output_mode=_internal_output_mode(config.concept_output_type),
        threshold_mode="auto",
        pooling="attn",
        group_unknown_threshold=0.50,
        validate=True,
    )
    detector.fit(noisy_data.training, noisy_data.validation)
    cbm = ConceptBasedModel(
        concept_detector=detector,
        label_predictor=FrontEndModel(),
        should_propagate=(config.concept_output_type == "continuous"),
    )
    cbm.label_predictor.fit(noisy_data.training.C, noisy_data.training.y)

    test_pred = cbm.predict(data.test)
    acc = float(np.mean(test_pred == data.test.y))
    logger.info("Subjective CBM Test Accuracy: %.4f", acc)

    save(cbm, config.get_model_path("cbm_subjective"), overwrite=True)
    return cbm


# ── Stage: train_dnn ─────────────────────────────────────────────────


def train_dnn(
    config: RobotBenchmarkConfig,
    data: ConceptDatasetSample | None = None,
) -> dict:
    """Fine-tune DistilBERT on text -> label (bypasses concepts).

    Returns a dict with accuracy metrics and model paths.
    """
    set_deterministic_seed(config.seed)
    device = determine_device()

    if data is None:
        data = load(config.get_dataset_path())

    acc, tok, model = _train_dnn_text(
        X_tr=data.training.X,
        y_tr=data.training.y,
        X_te=data.test.X,
        y_te=data.test.y,
        model_id=config.dnn_model_name,
        epochs=config.dnn_epochs,
        batch_size=config.dnn_batch_size,
        lr=config.dnn_lr,
        device=device,
    )

    # Platt calibration on validation
    calibrator = _fit_platt(data.validation.X, data.validation.y, tok, model, device)

    metrics = {"accuracy": acc, "seed": config.seed, "model": config.dnn_model_name}
    logger.info("DNN Test Accuracy: %.4f", acc)

    # Save model and metrics
    model_dir = config.get_model_path("dnn")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    save({"metrics": metrics, "calibrator": calibrator}, model_dir, overwrite=True)
    tok.save_pretrained(str(model_dir) + "_tok")
    model.save_pretrained(str(model_dir) + "_model")

    return metrics


# ── Stage: train_lfcbm ──────────────────────────────────────────────


def train_lfcbm(
    config: RobotBenchmarkConfig,
    data: ConceptDatasetSample | None = None,
) -> ConceptBasedModel:
    """Train label-free CBM using sentence embeddings + concepts CSV.

    Returns a ConceptBasedModel with LabelFreeDetector as concept source.
    """
    from concept_benchmark.synthetic.robot_text.lfcbm import LabelFreeDetector

    set_deterministic_seed(config.seed)

    if data is None:
        data = load(config.get_dataset_path())

    # Determine concepts CSV path
    concepts_csv = config.label_free_concepts_csv
    if not concepts_csv:
        default = (
            package_dir
            / "synthetic"
            / "helper"
            / "static"
            / "text_templates"
            / "concepts.csv"
        )
        if default.is_file():
            concepts_csv = str(default)

    lf_device = "cuda" if torch.cuda.is_available() else "cpu"
    lf_settings = {
        "concepts_csv": concepts_csv,
        "lf_alpha": 0.5,
        "lf_threshold": 0.5,
        "lf_mode": _internal_output_mode(config.concept_output_type),
        "lf_ridge": False,
        "lf_ridge_alpha": 1.0,
        "lf_encoder": config.label_free_encoder,
        "lf_device": lf_device,
        "lf_batch_size": 64,
        "lf_keep_k": 9,
        "lf_group_threshold": 0.9,
    }

    det_lf = LabelFreeDetector(lf_settings)
    det_lf.fit([str(x) for x in data.training.X], y=data.training.y.astype(int))

    # Use LFCBM predictions as concept features to train a student detector
    if config.concept_output_type == "binary":
        det_lf.settings["lf_mode"] = "hard"
    C_train = det_lf.predict([str(x) for x in data.training.X])

    # Build a CBM with a simple frontend trained on LFCBM concepts
    fe = FrontEndModel()
    fe.fit(C_train.astype(np.float32), data.training.y)

    # Wrap: we need a concept detector that can predict on a ConceptDatasetSample
    # For LFCBM, predictions are the LFCBM output; we create a thin wrapper
    detector = TextConceptDetector(
        epochs=config.detector_epochs,
        batch_size=config.detector_batch_size,
        lr=config.detector_lr,
        output_mode=_internal_output_mode(config.concept_output_type),
    )
    detector.fit(data.training, data.validation)

    cbm = ConceptBasedModel(
        concept_detector=detector,
        label_predictor=fe,
        should_propagate=(config.concept_output_type == "continuous"),
    )

    test_pred = cbm.predict(data.test)
    acc = float(np.mean(test_pred == data.test.y))
    logger.info("LFCBM Test Accuracy: %.4f", acc)

    save(cbm, config.get_model_path("lfcbm"), overwrite=True)
    det_lf.save(str(config.get_model_path("lfcbm")) + "_lf")
    return cbm
