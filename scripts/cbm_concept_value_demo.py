from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel

from concept_benchmark.data import ConceptDataset, ConceptDatasetSample
from concept_benchmark.models import ConceptDetector, ConceptBasedModel, FrontEndModel
from concept_benchmark.synthetic.robot_mm import create_multimodal_robot_dataset, load_concept_datasets, DEFAULT_CONCEPTS
from concept_benchmark.synthetic.robot_text import TextConceptDataset
from concept_benchmark.synthetic.helper.text_concept_detector import TextConceptDetector
from concept_benchmark.metrics import calc_metric


settings = {
    "out_dir": "results/concept_value_demo",
    "mode": "incomplete_union",
    "n": 800,
    "image_size": 256,
    "color_mode": "color",
    "missing_rate": 0.35,
    "p_overlap": 0.3,
    "seed": 0,
    "concept_mode": "hard",
    "human_label_frac": 0.2,
    "intervention_k": 2,
    "tau": 0.5,
    "clip_model": "openai/clip-vit-base-patch32",
    "machine_method": "clip",
    "eval_split": "test",
}


def _read_pairs(meta_json: Path):
    m = json.loads(Path(meta_json).read_text())

    pairs = pd.read_csv(m["pairs_csv"])

    C = np.vstack(pairs["C_true"].apply(json.loads).tolist()).astype(int)
    mask_img = np.vstack(pairs["mask_img"].apply(lambda s: list(json.loads(s).values())).tolist()).astype(bool)
    mask_txt = np.vstack(pairs["mask_text"].apply(lambda s: list(json.loads(s).values())).tolist()).astype(bool)

    return C, mask_img, mask_txt, m


def _split(ds: ConceptDataset, seed: int = 0):
    ds.generate_cvindices(seed=seed)
    ds.split("K05N01", fold_num_validation=4, fold_num_test=5)

    return ds.training, ds.validation, ds.test


def _onehot_names(concepts: dict[str, list[str]]):
    names = []

    for k, vals in concepts.items():
        lv = set(map(str.lower, vals))

        if lv == {"true", "false"}:
            names.append(k)

        elif len(vals) == 2:
            names.append(f"{k}={vals[0]}")

        else:
            for v in vals:
                names.append(f"{k}={v}")

    return names


class OracleConceptDetector(ConceptDetector):
    def __init__(self, C_all: np.ndarray):
        super().__init__(embedding_model=None)
        self._C = np.asarray(C_all, dtype=float)

    def fit(self, *args, **kwargs):
        return self

    def predict(self, dataset: ConceptDatasetSample, **kwargs) -> np.ndarray:
        idx = np.where(dataset.indices)[0] if dataset.indices is not None else np.arange(len(dataset))

        return self._C[idx].astype(np.float32)


def _concept_prompts(names: list[str]):
    out = []

    for n in names:
        if "=" in n:
            k, v = n.split("=", 1)

            if k == "ears_shape" and v == "triangle":
                out.append("a robot with triangular ears")

            elif k == "mouth_type":
                out.append(f"a robot with {v} mouth")

            elif k == "hand_shape":
                vv = v.replace("_", " ")
                out.append(f"a robot with {vv} hands")

            elif k == "foot_shape":
                vv = v.replace("_", " ")
                out.append(f"a robot with {vv} feet")

            else:
                out.append(f"a robot with {v} {k.replace('_', ' ')}")

        else:
            if n.startswith("has_"):
                out.append("a robot with " + n[4:].replace("_", " "))
            else:
                out.append("a robot with " + n.replace("_", " "))

    return out


class ClipConceptDetector(ConceptDetector):
    def __init__(
        self,
        image_dir: Path,
        concept_names: list[str],
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        super().__init__(embedding_model=None)

        self.image_dir = Path(image_dir)
        self.names = concept_names
        self.prompts = _concept_prompts(concept_names)

        self.device = (
            device
            or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        )

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model_name)

    def fit(self, *args, **kwargs):
        return self

    def predict(self, dataset: ConceptDatasetSample, **kwargs) -> np.ndarray:
        from PIL import Image

        paths = [self.image_dir / x if isinstance(x, str) else x for x in dataset.X]
        texts = self.prompts

        cap = 64
        out = []

        with torch.no_grad():
            for s in range(0, len(paths), cap):
                ims = [Image.open(p).convert("RGB") for p in paths[s : s + cap]]

                enc = self.proc(text=texts, images=ims, return_tensors="pt", padding=True).to(self.device)

                feats = self.model(**enc)

                it = feats.logits_per_image
                it = it - it.mean(dim=1, keepdim=True)
                it = it / (it.std(dim=1, keepdim=True) + 1e-6)

                pr = torch.sigmoid(it).cpu().numpy()
                out.append(pr)

        return np.vstack(out).astype(np.float32)


class AgopConceptDetector(ConceptDetector):
    def __init__(self, image_dir: Path, concept_dim: int = 8, device: str | None = None):
        super().__init__(embedding_model=None)

        self.image_dir = Path(image_dir)
        self.k = int(concept_dim)

        self.device = (
            device
            or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        )

        self.encoder = torch.hub.load("pytorch/vision:v0.14.0", "vit_b_16", pretrained=True).to(self.device).eval()
        self.head = nn.Linear(768, 2).to(self.device)

    def fit(self, train_dataset: ConceptDatasetSample, valid_dataset: ConceptDatasetSample, **kwargs):
        from PIL import Image

        tf = kwargs.get("tf", None)
        opt = torch.optim.Adam(self.head.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(5):
            xb = []
            yb = []

            for i in np.random.permutation(len(train_dataset))[:512]:
                p = self.image_dir / train_dataset.X[i]

                x = Image.open(p).convert("RGB")

                if tf:
                    x = tf(x)
                else:
                    from torchvision import transforms

                    x = transforms.Resize((224, 224))(x)
                    x = transforms.ToTensor()(x)

                xb.append(x)
                yb.append(int(train_dataset.y[i]))

            xb = torch.stack(xb).to(self.device)
            yb = torch.tensor(yb, dtype=torch.long, device=self.device)

            with torch.no_grad():
                z = self.encoder(xb)

            logits = self.head(z)

            loss = nn.CrossEntropyLoss()(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        return self

    def predict(self, dataset: ConceptDatasetSample, **kwargs) -> np.ndarray:
        from PIL import Image

        tf = kwargs.get("tf", None)

        cap = 64
        xs = []

        paths = [self.image_dir / x for x in dataset.X]

        with torch.no_grad():
            for s in range(0, len(paths), cap):
                ims = []

                for p in paths[s : s + cap]:
                    x = Image.open(p).convert("RGB")

                    if tf:
                        x = tf(x)
                    else:
                        from torchvision import transforms

                        x = transforms.Resize((224, 224))(x)
                        x = transforms.ToTensor()(x)

                    ims.append(x)

                xb = torch.stack(ims).to(self.device)

                z = self.encoder(xb).detach().cpu().numpy()

                xs.append(z)

        H = np.vstack(xs)
        G = H.T @ H / float(H.shape[0])

        w, V = np.linalg.eigh(G + 1e-6 * np.eye(G.shape[0]))
        V = V[:, -self.k :]

        Z = H @ V
        Z = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-6)

        P = 1.0 / (1.0 + np.exp(-Z))

        return P.astype(np.float32)


class HumanImageConceptDetector(ConceptDetector):
    def __init__(self, image_dir: Path, concept_dim: int, device: str | None = None):
        super().__init__(embedding_model=None)

        self.image_dir = Path(image_dir)
        self.c = int(concept_dim)

        self.device = (
            device
            or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        )

        self.encoder = torch.hub.load("pytorch/vision:v0.14.0", "vit_b_16", pretrained=True).to(self.device).eval()
        self.head = nn.Linear(768, self.c).to(self.device)

    def fit(self, train_dataset: ConceptDatasetSample, valid_dataset: ConceptDatasetSample, **kwargs):
        from PIL import Image

        tf = kwargs.get("tf", None)

        opt = torch.optim.Adam(self.head.parameters(), lr=1e-3, weight_decay=1e-4)

        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        obs = train_dataset.meta.get("observed_mask", None)

        if obs is None:
            obs = np.ones((len(train_dataset), self.c), dtype=np.float32)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        for _ in range(6):
            idx = np.random.permutation(len(train_dataset))[:512]

            ims = []
            ys = []
            ms = []

            for i in idx:
                p = self.image_dir / train_dataset.X[i]

                x = Image.open(p).convert("RGB")

                if tf:
                    x = tf(x)
                else:
                    from torchvision import transforms

                    x = transforms.Resize((224, 224))(x)
                    x = transforms.ToTensor()(x)

                ims.append(x)
                ys.append(train_dataset.C[i])
                ms.append(obs[i].cpu().numpy())

            xb = torch.stack(ims).to(self.device)
            yb = torch.tensor(np.asarray(ys), dtype=torch.float32, device=self.device)
            mb = torch.tensor(np.asarray(ms), dtype=torch.float32, device=self.device)

            with torch.no_grad():
                z = self.encoder(xb)

            logits = self.head(z)

            loss_all = loss_fn(logits, yb)
            loss = (loss_all * mb).sum() / torch.clamp(mb.sum(), min=1.0)

            opt.zero_grad()
            loss.backward()
            opt.step()

        return self

    def predict(self, dataset: ConceptDatasetSample, **kwargs) -> np.ndarray:
        from PIL import Image

        tf = kwargs.get("tf", None)

        cap = 64
        out = []

        paths = [self.image_dir / x for x in dataset.X]

        with torch.no_grad():
            for s in range(0, len(paths), cap):
                ims = []

                for p in paths[s : s + cap]:
                    x = Image.open(p).convert("RGB")

                    if tf:
                        x = tf(x)
                    else:
                        from torchvision import transforms

                        x = transforms.Resize((224, 224))(x)
                        x = transforms.ToTensor()(x)

                    ims.append(x)

                xb = torch.stack(ims).to(self.device)

                z = self.encoder(xb)
                logits = self.head(z)

                pr = torch.sigmoid(logits).cpu().numpy()

                out.append(pr)

        return np.vstack(out).astype(np.float32)


def _augment_with_irrelevant(C: np.ndarray, names: list[str], meta: dict) -> tuple[np.ndarray, list[str]]:
    df_img = pd.read_csv(meta["image_csv"])

    if "left_color" in df_img.columns and "right_color" in df_img.columns:
        lc = df_img["left_color"].apply(json.loads).to_numpy()
        rc = df_img["right_color"].apply(json.loads).to_numpy()

        l_dark = np.array([1 if sum(v) < 300 else 0 for v in lc], dtype=int).reshape(-1, 1)
        r_dark = np.array([1 if sum(v) < 300 else 0 for v in rc], dtype=int).reshape(-1, 1)

    else:
        rng = np.random.default_rng(0)

        l_dark = rng.integers(0, 2, size=(C.shape[0], 1))
        r_dark = rng.integers(0, 2, size=(C.shape[0], 1))

    rng = np.random.default_rng(1)

    noise = rng.integers(0, 2, size=(C.shape[0], 1))

    C2 = np.concatenate([C, l_dark, r_dark, noise], axis=1)
    names2 = names + ["left_color_dark", "right_color_dark", "noise_bit"]

    return C2, names2


def _to_text_ds(names: list[str], classes: list[str], df_txt: pd.DataFrame, C_true: np.ndarray, mask_txt: np.ndarray):
    X = df_txt["text"].tolist()
    y = df_txt["y"].values.astype(int)
    obs = mask_txt.astype(int)

    meta_txt = {"classes": classes, "concepts": names, "data_type": "text", "observed_mask": obs}

    return TextConceptDataset(X=X, C=C_true, y=y, meta=meta_txt)


def _to_image_ds(
    names: list[str],
    classes: list[str],
    X_paths: list[str],
    y: np.ndarray,
    C_true: np.ndarray,
    mask_img: np.ndarray,
    image_dir: str,
):
    obs = mask_img.astype(int)

    meta_img = {
        "classes": classes,
        "concepts": names,
        "data_type": "image",
        "observed_mask": obs,
        "image_dir": image_dir,
    }

    return ConceptDatasetSample(parent=None, X=X_paths, C=C_true, y=y, meta=meta_img)


def _front_end():
    return FrontEndModel()


def _eval_cbm(
    name: str,
    ds_train: ConceptDatasetSample,
    ds_val: ConceptDatasetSample,
    ds_test: ConceptDatasetSample,
    det: ConceptDetector,
    tau: float = 0.5,
    propagate: bool = True,
):
    m = ConceptBasedModel(concept_detector=det, front_end_model=_front_end(), propagate=propagate)

    m.fit(ds_train, ds_val, freeze=True, fit_params={"epochs": 8, "device": "cpu", "batch_size": 64})

    pr = m.predict_proba(ds_test)
    yhat = np.argmax(pr, axis=1)

    acc = float((yhat == ds_test.y).mean())

    abst = calc_metric(pr.max(axis=1), (ds_test.y == yhat).astype(int), tau=tau)

    return {"name": name, "acc": acc, "coverage": abst["coverage"], "selective_accuracy": abst["selective_accuracy"]}


def _intervene_eval(m: ConceptBasedModel, ds: ConceptDatasetSample, C_true: np.ndarray, k: int):
    with torch.no_grad():
        pr_c = m.concept_detector.predict(ds)

    idx = np.where(ds.indices)[0] if ds.indices is not None else np.arange(len(ds))

    Cp = pr_c.copy()

    for i in range(len(ds)):
        j = np.argsort(np.abs(Cp[i] - C_true[idx[i]]))[:k]
        Cp[i, j] = C_true[idx[i], j]

    y0 = m.front_end_model.predict_proba(pr_c)
    y1 = m.front_end_model.predict_proba(Cp)

    a0 = float((np.argmax(y0, 1) == ds.y).mean())
    a1 = float((np.argmax(y1, 1) == ds.y).mean())

    return {"pre": a0, "post": a1, "delta": a1 - a0}


def _relevant_indices(names: list[str]):
    idx = []

    for i, n in enumerate(names):
        if n == "body_shape=square":
            idx.append(i)

        if n.startswith("foot_shape=pointy_"):
            idx.append(i)

    return idx


def _subset_mask(mask: np.ndarray, names: list[str]):
    idx = _relevant_indices(names)

    if not idx:
        return np.zeros(mask.shape[0], dtype=bool)

    return (~mask[:, idx]).all(axis=1)


def _subset_acc(model: ConceptBasedModel, ds: ConceptDatasetSample, mask_rows: np.ndarray):
    pr = model.predict_proba(ds)
    yhat = np.argmax(pr, axis=1)

    m = mask_rows[: len(yhat)]

    if not np.any(m):
        return float("nan")

    return float((yhat[m] == ds.y[m]).mean())


def run():
    ap = argparse.ArgumentParser(add_help=False)

    ap.add_argument("--out-dir", type=str, default=settings["out_dir"])
    ap.add_argument("--mode", choices=["complete_both", "complete_union", "incomplete_union"], default=settings["mode"])
    ap.add_argument("--n", type=int, default=settings["n"])
    ap.add_argument("--image-size", type=int, default=settings["image_size"])
    ap.add_argument("--color-mode", choices=["color", "greyscale"], default=settings["color_mode"])
    ap.add_argument("--missing-rate", type=float, default=settings["missing_rate"])
    ap.add_argument("--p-overlap", type=float, default=settings["p_overlap"])
    ap.add_argument("--seed", type=int, default=settings["seed"])
    ap.add_argument("--human-label-frac", type=float, default=settings["human_label_frac"])
    ap.add_argument("--intervention-k", type=int, default=settings["intervention_k"])
    ap.add_argument("--tau", type=float, default=settings["tau"])
    ap.add_argument("--clip-model", type=str, default=settings["clip_model"])
    ap.add_argument("--machine-method", choices=["clip", "agop"], default=settings["machine_method"])
    ap.add_argument("--eval-split", choices=["validation", "test"], default=settings["eval_split"])
    ap.add_argument("--include-irrelevant", action="store_true", default=False)

    args, _ = ap.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = create_multimodal_robot_dataset(
        mode=args.mode,
        n=args.n,
        concepts=DEFAULT_CONCEPTS,
        seed=args.seed,
        out_dir=str(out_dir),
        image_size=args.image_size,
        color_mode=args.color_mode,
        missing_rate=args.missing_rate,
        p_overlap=args.p_overlap,
    )

    C_true, mask_img, mask_txt, meta = _read_pairs(Path(out.meta_json))
    names = meta["concept_names"]

    if args.include_irrelevant:
        C_true, names = _augment_with_irrelevant(C_true, names, meta)

        mask_img = np.hstack([mask_img, np.zeros((mask_img.shape[0], 3), dtype=bool)])
        mask_txt = np.hstack([mask_txt, np.zeros((mask_txt.shape[0], 3), dtype=bool)])

    ds_img, ds_txt = load_concept_datasets(out.meta_json)

    tr_i, va_i, te_i = _split(ds_img, seed=args.seed)
    tr_t, va_t, te_t = _split(ds_txt, seed=args.seed)

    df_text = pd.read_csv(meta["text_csv"])

    txt_train = _to_text_ds(
        names,
        meta.get("classes", ["drent", "glorp"]),
        df_text.iloc[np.where(tr_t.indices)[0]],
        C_true[np.where(tr_t.indices)[0]],
        mask_txt[np.where(tr_t.indices)[0]],
    )

    txt_valid = _to_text_ds(
        names,
        meta.get("classes", ["drent", "glorp"]),
        df_text.iloc[np.where(va_t.indices)[0]],
        C_true[np.where(va_t.indices)[0]],
        mask_txt[np.where(va_t.indices)[0]],
    )

    txt_test = _to_text_ds(
        names,
        meta.get("classes", ["drent", "glorp"]),
        df_text.iloc[np.where(te_t.indices)[0]],
        C_true[np.where(te_t.indices)[0]],
        mask_txt[np.where(te_t.indices)[0]],
    )

    frac = float(args.human_label_frac)

    if frac < 1.0:
        rng = np.random.default_rng(args.seed)

        obs = txt_train.meta["observed_mask"].copy()

        n_all = obs.shape[0] * obs.shape[1]
        k = int((1.0 - frac) * n_all)

        if k > 0:
            idx = rng.choice(n_all, size=k, replace=False)

            r = idx // obs.shape[1]
            c = idx % obs.shape[1]

            obs[r, c] = 0

        txt_train.meta["observed_mask"] = obs

    idx_tr = np.where(tr_i.indices)[0]
    idx_va = np.where(va_i.indices)[0]
    idx_te = np.where(te_i.indices)[0]

    img_train = _to_image_ds(
        names,
        meta["classes"],
        tr_i.X,
        tr_i.y,
        C_true[idx_tr],
        mask_img[idx_tr],
        meta["image_dir"],
    )

    img_valid = _to_image_ds(
        names,
        meta["classes"],
        va_i.X,
        va_i.y,
        C_true[idx_va],
        mask_img[idx_va],
        meta["image_dir"],
    )

    img_test = _to_image_ds(
        names,
        meta["classes"],
        te_i.X,
        te_i.y,
        C_true[idx_te],
        mask_img[idx_te],
        meta["image_dir"],
    )

    if frac < 1.0:
        rng = np.random.default_rng(args.seed + 1)

        obs_i = img_train.meta["observed_mask"].copy()

        n_all_i = obs_i.shape[0] * obs_i.shape[1]
        k_i = int((1.0 - frac) * n_all_i)

        if k_i > 0:
            idx = rng.choice(n_all_i, size=k_i, replace=False)

            r = idx // obs_i.shape[1]
            c = idx % obs_i.shape[1]

            obs_i[r, c] = 0

        img_train.meta["observed_mask"] = obs_i

    gt = OracleConceptDetector(C_true)

    human_txt = TextConceptDetector(output_mode="hard")
    human_txt.fit(txt_train, txt_valid, device="cpu", epochs=6, batch_size=64)

    human_img = HumanImageConceptDetector(image_dir=Path(meta["image_dir"]), concept_dim=len(names))
    human_img.fit(img_train, img_valid)

    if args.machine_method == "clip":
        mach = ClipConceptDetector(
            image_dir=Path(meta["image_dir"]),
            concept_names=names,
            model_name=args.clip_model,
        )

        mach.fit(None, None)

    else:
        mach = AgopConceptDetector(image_dir=Path(meta["image_dir"]), concept_dim=8)
        mach.fit(tr_i, va_i)

    res = []

    r_gt = _eval_cbm("ground_truth", img_train, img_valid, img_test, gt, tau=args.tau, propagate=True)
    r_hu_txt = _eval_cbm("human_text", txt_train, txt_valid, txt_test, human_txt, tau=args.tau, propagate=True)
    r_hu_img = _eval_cbm("human_image", img_train, img_valid, img_test, human_img, tau=args.tau, propagate=True)

    r_mc = _eval_cbm("machine_" + args.machine_method, img_train, img_valid, img_test, mach, tau=args.tau, propagate=True)

    res.extend([r_gt, r_hu_txt, r_hu_img, r_mc])

    m_gt = ConceptBasedModel(concept_detector=gt, front_end_model=_front_end(), propagate=True)
    m_gt.fit(img_train, img_valid, freeze=True, fit_params={"epochs": 8, "device": "cpu"})

    m_hu_txt = ConceptBasedModel(concept_detector=human_txt, front_end_model=_front_end(), propagate=True)
    m_hu_txt.fit(txt_train, txt_valid, freeze=True, fit_params={"epochs": 8, "device": "cpu"})

    m_hu_img = ConceptBasedModel(concept_detector=human_img, front_end_model=_front_end(), propagate=True)
    m_hu_img.fit(img_train, img_valid, freeze=True, fit_params={"epochs": 8, "device": "cpu"})

    m_mc = ConceptBasedModel(concept_detector=mach, front_end_model=_front_end(), propagate=True)
    m_mc.fit(img_train, img_valid, freeze=True, fit_params={"epochs": 8, "device": "cpu"})

    inter_gt = _intervene_eval(m_gt, img_test, C_true, k=int(args.intervention_k))
    inter_hu_txt = _intervene_eval(m_hu_txt, txt_test, C_true, k=int(args.intervention_k))
    inter_hu_img = _intervene_eval(m_hu_img, img_test, C_true, k=int(args.intervention_k))
    inter_mc = _intervene_eval(m_mc, img_test, C_true, k=int(args.intervention_k))

    idx_te_all = np.where(te_i.indices)[0]

    sub_img = _subset_mask(mask_img[idx_te_all], names)
    sub_txt = _subset_mask(mask_txt[idx_te_all], names)

    robust = {
        "ground_truth_on_img_missing": _subset_acc(m_gt, img_test, sub_img),
        "human_text_on_text_missing": _subset_acc(m_hu_txt, txt_test, sub_txt),
        "human_image_on_img_missing": _subset_acc(m_hu_img, img_test, sub_img),
        "machine_{}_on_img_missing".format(args.machine_method): _subset_acc(m_mc, img_test, sub_img),
    }

    out_json = {
        "meta": meta,
        "settings": vars(args),
        "results": res,
        "interventions": {
            "ground_truth": inter_gt,
            "human_text": inter_hu_txt,
            "human_image": inter_hu_img,
            "machine_" + args.machine_method: inter_mc,
        },
        "robustness": robust,
    }

    (out_dir / "summary.json").write_text(json.dumps(out_json, indent=2))

    print(json.dumps(out_json, indent=2))


run()
