import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def _idx(names):
    return {n: i for i, n in enumerate(names)}

def _mask_sets(names):
    ni = _idx(names)
    mouth = [ni[n] for n in names if ("mouth" in n)]
    coarse = [ni[n] for n in names if n in {"foot_shape", "foot_shape_pointy", "foot_shape_flat"}]
    p_sub = [ni[n] for n in names if any(k in n for k in ["p3sid","p4sid","p5sid","psq","ptrap","prnd","pointy_","foot_shape_pointy_"])]
    f_sub = [ni[n] for n in names if any(k in n for k in ["f5sid","fsq","ftrap","frnd","flshp","f_trap","flat_","foot_shape_flat_"])]
    sub = sorted(set(p_sub + f_sub))
    return {"mouth": mouth, "foot_coarse": coarse, "foot_pointy_sub": p_sub, "foot_flat_sub": f_sub, "foot_sub": sub}

def _or_prob(p, axes):
    if len(axes) == 0: return None
    q = 1.0
    for a in axes: q *= (1.0 - p[:, a])
    return 1.0 - q

def _or_bool(C, axes):
    if len(axes) == 0: return None
    X = (C[:, axes] > 0).astype(np.float32)
    return (X.max(axis=1)).astype(np.float32)

def _coarse_from_C(C, names, masks):
    Xm = None
    if len(masks["mouth"]) > 0:
        Xm = C[:, masks["mouth"]].astype(np.float32)
        if Xm.ndim == 1: Xm = Xm[:, None]
    if len(masks["foot_coarse"]) > 0:
        Xf = C[:, masks["foot_coarse"]].astype(np.float32)
        if Xf.ndim == 1: Xf = Xf[:, None]
        if Xf.shape[1] > 1: Xf = Xf[:, :1]
    else:
        Xf_or = _or_bool(C, masks["foot_pointy_sub"]); Xf = Xf_or[:, None] if Xf_or is not None else None
    parts = []; 
    if Xm is not None: parts.append(Xm)
    if Xf is not None: parts.append(Xf)
    if not parts: return None, None
    X = np.concatenate(parts, axis=1)
    feats = (["mouth"] if Xm is not None else []) + (["foot_shape_pointy"] if Xf is not None else [])
    return X, feats

def _coarse_from_P(P, names, masks):
    Xm = None
    if len(masks["mouth"]) > 0:
        Xm = P[:, masks["mouth"]].astype(np.float32)
        if Xm.ndim == 1: Xm = Xm[:, None]
    if len(masks["foot_coarse"]) > 0:
        Xf = P[:, masks["foot_coarse"]].astype(np.float32)
        if Xf.ndim == 1: Xf = Xf[:, None]
        if Xf.shape[1] > 1: Xf = Xf[:, :1]
    else:
        Xf_or = _or_prob(P, masks["foot_pointy_sub"]); Xf = Xf_or[:, None] if Xf_or is not None else None
    parts = []
    if Xm is not None: parts.append(Xm)
    if Xf is not None: parts.append(Xf)
    if not parts: return None, None
    X = np.concatenate(parts, axis=1)
    feats = (["mouth"] if Xm is not None else []) + (["foot_shape_pointy"] if Xf is not None else [])
    return X, feats

def _sub_from_C(C, names, masks):
    Xm = None
    if len(masks["mouth"]) > 0:
        Xm = C[:, masks["mouth"]].astype(np.float32)
        if Xm.ndim == 1: Xm = Xm[:, None]
    Xf = C[:, masks["foot_sub"]].astype(np.float32) if len(masks["foot_sub"]) > 0 else None
    parts, fn = [], []
    if Xm is not None: parts.append(Xm); fn += ["mouth"]
    if Xf is not None: parts.append(Xf); fn += [names[j] for j in masks["foot_sub"]]
    if not parts: return None, None
    return np.concatenate(parts, axis=1), fn

def _sub_from_P(P, names, masks):
    Xm = None
    if len(masks["mouth"]) > 0:
        Xm = P[:, masks["mouth"]].astype(np.float32)
        if Xm.ndim == 1: Xm = Xm[:, None]
    Xf = P[:, masks["foot_sub"]].astype(np.float32) if len(masks["foot_sub"]) > 0 else None
    parts, fn = [], []
    if Xm is not None: parts.append(Xm); fn += ["mouth"]
    if Xf is not None: parts.append(Xf); fn += [names[j] for j in masks["foot_sub"]]
    if not parts: return None, None
    return np.concatenate(parts, axis=1), fn

def _fit_lr(X, y):
    m = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=200)
    m.fit(X, y); return m

def _acc(m, X, y):
    return float(accuracy_score(y, m.predict(X)))

def _subset(X, feats, which):
    if X is None: return None, []
    if which == "all": return X, feats
    idx = [i for i, n in enumerate(feats) if (("mouth" in n) if which == "mouth" else ("mouth" not in n))]
    if not idx: return None, []
    return X[:, idx], [feats[i] for i in idx]

def _dump_weights(m, feats, k=10):
    W = m.coef_.ravel(); order = np.argsort(-np.abs(W))
    return [(feats[i], float(W[i])) for i in order[: min(k, len(order))]]

def run_fe_harness(C_tr, y_tr, C_te, y_te, P_tr, P_te, concept_names, table_name="FE 2x2"):
    masks = _mask_sets(concept_names)
    Xcc_tr, fcc = _coarse_from_C(C_tr, concept_names, masks); Xcc_te, _ = _coarse_from_C(C_te, concept_names, masks)
    Xcp_tr, fcp = _coarse_from_P(P_tr, concept_names, masks); Xcp_te, _ = _coarse_from_P(P_te, concept_names, masks)
    Xsc_tr, fsc = _sub_from_C(C_tr, concept_names, masks);   Xsc_te, _ = _sub_from_C(C_te, concept_names, masks)
    Xsp_tr, fsp = _sub_from_P(P_tr, concept_names, masks);   Xsp_te, _ = _sub_from_P(P_te, concept_names, masks)
    cells = {("coarse","C"):(Xcc_tr,fcc,Xcc_te),("coarse","P"):(Xcp_tr,fcp,Xcp_te),("sub","C"):(Xsc_tr,fsc,Xsc_te),("sub","P"):(Xsp_tr,fsp,Xsp_te)}
    print(f"\n{table_name}"); print("mode\tfeat\tacc")
    weight_dump = {}
    for mode in [("coarse","C"),("coarse","P"),("sub","C"),("sub","P")]:
        Xtr, feats, Xte = cells[mode]
        for which in ["all","feet","mouth"]:
            Xtr_sub, feats_sub = _subset(Xtr, feats, which); Xte_sub, _ = _subset(Xte, feats, which)
            if Xtr_sub is None or Xte_sub is None: print(f"{mode[0]}-{mode[1]}\t{which}\tNA"); continue
            m = _fit_lr(Xtr_sub, y_tr); a = _acc(m, Xte_sub, y_te); print(f"{mode[0]}-{mode[1]}\t{which}\t{a:.4f}")
            if which == "all": weight_dump[mode] = _dump_weights(m, feats_sub, k=10)
    print("\nweights (top |w|) for 'all' in each cell:")
    for mode in [("coarse","C"),("coarse","P"),("sub","C"),("sub","P")]:
        tops = weight_dump.get(mode, []); tag = f"{mode[0]}-{mode[1]}"
        s = ", ".join([f"{n}:{w:+.3f}" for n, w in tops]); print(f"{tag}: {s}")