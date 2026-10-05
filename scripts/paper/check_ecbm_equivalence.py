"""Check that experiments/baselines/ecbm.py reproduces the authors' ECBM (github.com/xmed-lab/ECBM).

Loads the authors' `EBM_GL` (networks/EBM.py) and `EarlyStopping` (utils.py) from a checkout, copies its weights
into our `_ECBMNet` (backbone = identity, so both see the same features), and compares on random inputs:
  1. training energies (concept augmentation switched off in both, since it is random);
  2. inference energies for given label/concept variables;
  3. the label after gradient inference without interventions (weights 1/1/0.01) and after an intervention
     (weights 0/0/3), running the authors' `run_optim` loop (GradientInference.py:39-71) on their module.

    PYTHONPATH=. python scripts/paper/check_ecbm_equivalence.py --original /tmp/ecbm_orig
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

from experiments.baselines import ecbm as ours


def load_original(root: Path):
    """Import networks/EBM.py without its torchvision backbone, and utils.EarlyStopping."""
    package = types.ModuleType("networks")
    package.__path__ = [str(root / "networks")]
    sys.modules["networks"] = package
    backbone = types.ModuleType("networks.backbone")
    backbone.get_model = lambda *a, **k: nn.Identity()
    sys.modules["networks.backbone"] = backbone
    spec = importlib.util.spec_from_file_location(
        "networks.EBM", root / "networks/EBM.py"
    )
    ebm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ebm)
    import numpy

    numpy.Inf = numpy.inf  # utils.py predates NumPy 2
    sys.path.insert(0, str(root))  # utils.py imports the authors' metrics.py
    spec = importlib.util.spec_from_file_location("ecbm_utils", root / "utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return ebm, utils


def run_optim_original(module, utils, x, y_prob, c_prob, weights, patience=10):
    """The authors' run_optim (GradientInference.py:39-71) on EBM_GL; returns softmaxed label variable."""
    lambda_xy, lambda_xc, lambda_cy = weights
    module.y_prob = nn.Parameter(y_prob.clone())
    module.c_prob = nn.Parameter(c_prob.clone())
    optim = torch.optim.Adam(
        [{"params": [module.c_prob], "lr": 0.1}, {"params": [module.y_prob], "lr": 0.1}]
    )
    stop = utils.EarlyStopping(patience=patience)
    with torch.enable_grad():
        running = True
        while running:
            module.eval()
            optim.zero_grad()
            xy_en, cy_en, c_en, prob = module(x, None, False, use_cy=True)
            cpt_loss = torch.zeros([])
            for i in range(c_en.shape[1]):
                cpt_loss += c_en[:, i, :].mean()
            loss = (
                lambda_xy * xy_en.mean()
                + lambda_xc * cpt_loss
                + lambda_cy * cy_en.mean()
            )
            stop(xy_en.mean(), None)
            loss.backward(retain_graph=True)
            optim.step()
            if stop.early_stop:
                running = False
    return torch.softmax(module.y_prob.detach().squeeze(-1), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    ebm, utils = load_original(args.original)
    n_classes, n_concepts, hid, feat, bs = 2, 7, 64, 128, 32
    original = ebm.EBM_GL(
        Namespace(cy_perturb_prob=0.2, cy_permute_prob=0.2),
        num_classes=n_classes,
        input_size=feat,
        hid_size=hid,
        cpt_size=n_concepts,
    )
    original.cy_augment = lambda c_gt, permute_ratio, permute_prob=0.2: (
        c_gt
    )  # augmentation off for the comparison
    port = ours._ECBMNet(
        n_concepts=n_concepts,
        n_tasks=n_classes,
        hid_size=hid,
        feature_dim=feat,
        lambda_xy=3,
        lambda_xc=1,
        lambda_cy=1,
        c_extractor_arch=lambda d: nn.Identity(),
    )
    missing, unexpected = port.load_state_dict(
        {
            k: v
            for k, v in original.state_dict().items()
            if k not in ("y_prob", "c_prob", "fc_c.weight", "fc_c.bias")
        },
        strict=False,
    )
    assert not unexpected and all(k.startswith("backbone") for k in missing), (
        missing,
        unexpected,
    )
    original.eval()
    port.eval()

    x = torch.randn(bs, feat)
    c = (torch.rand(bs, n_concepts) > 0.5).float()
    checks = []

    o_xy, o_cy, o_xc = original(x, c, True, use_cy=True)
    p_xy, p_cy, p_xc = port.training_energies(x, c, augment=False)
    checks += [
        ("training xy", o_xy, p_xy),
        ("training cy", o_cy, p_cy),
        ("training xc", o_xc, p_xc),
    ]

    spec = importlib.util.spec_from_file_location(
        "ecbm_loss", args.original / "loss.py"
    )
    loss_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loss_mod)
    y = torch.randint(0, n_classes, (bs,))
    o_label = loss_mod.EBMLoss_label(list(range(n_classes)), device="cpu")
    o_concept = loss_mod.EBMLoss_concept(list(range(n_concepts)), device="cpu")
    o_loss = (
        3 * o_label(o_xy, y) + 1 * o_concept(o_xc, c) + 1 * o_label(o_cy, y)
    )  # LitModel.training_step
    original_augment = ours._flip_concepts
    ours._flip_concepts = lambda present: present
    p_loss, _ = ours._ecbm_losses(port, x, c, y)
    ours._flip_concepts = original_augment
    checks.append(("training loss", o_loss, p_loss))

    y_var = torch.randn(bs, n_classes, 1)
    c_var = torch.randn(bs, n_concepts, 2)
    original.y_prob = nn.Parameter(y_var.clone())
    original.c_prob = nn.Parameter(c_var.clone())
    o_xy, o_cy, o_xc, _ = original(x, None, False, use_cy=True)
    p_xy, p_cy, p_xc = port.inference_energies(x, y_var.squeeze(-1), c_var)
    checks += [
        ("inference xy", o_xy.view(-1), p_xy),
        ("inference cy", o_cy.view(-1), p_cy),
        ("inference xc", o_xc.squeeze(-1), p_xc),
    ]

    o_y = run_optim_original(
        original,
        utils,
        x,
        torch.zeros(bs, n_classes, 1),
        torch.zeros(bs, n_concepts, 2),
        ours.INFERENCE_WEIGHTS,
    )
    o_c_logits = original.c_prob.detach().clone()
    p_y_logits, p_c_logits = ours._infer_labels_and_concepts(port, x)
    checks += [
        ("inference label", o_y, torch.softmax(p_y_logits, -1)),
        ("inference concepts", o_c_logits, p_c_logits),
    ]

    mask = torch.rand(bs, n_concepts) > 0.5
    forced = (torch.nn.functional.one_hot(c.long(), 2).float() - 0.5) * 10
    o_y = run_optim_original(
        original,
        utils,
        x,
        torch.zeros(bs, n_classes, 1),
        torch.where(mask.unsqueeze(-1), forced, o_c_logits),
        ours.INTERVENTION_WEIGHTS,
    )
    p_y = torch.softmax(ours._intervene(port, x, p_c_logits, c, mask), -1)
    checks.append(("intervention label", o_y, p_y))

    # Running many test batches side by side must match running them one after another.
    xs = torch.randn(150, feat)
    serial = torch.cat(
        [
            ours._infer_labels_and_concepts(port, xs[i : i + 32])[0]
            for i in range(0, 150, 32)
        ]
    )
    parallel = ours._infer_labels_and_concepts(port, xs, batch_size=32)[0]
    checks.append(
        ("batched inference", torch.softmax(serial, -1), torch.softmax(parallel, -1))
    )
    cs = (torch.rand(150, n_concepts) > 0.5).float()
    ms = torch.rand(150, n_concepts) > 0.5
    cl = torch.randn(150, n_concepts, 2)
    serial = torch.cat(
        [
            ours._intervene(
                port, xs[i : i + 32], cl[i : i + 32], cs[i : i + 32], ms[i : i + 32]
            )
            for i in range(0, 150, 32)
        ]
    )
    parallel = ours._intervene(port, xs, cl, cs, ms, batch_size=32)
    checks.append(
        ("batched intervention", torch.softmax(serial, -1), torch.softmax(parallel, -1))
    )

    worst = 0.0
    for name, a, b in checks:
        diff = float((a.detach() - b.detach()).abs().max())
        worst = max(worst, diff)
        print(f"{name:20} max |diff| = {diff:.2e}")
    assert worst < 1e-4, f"port differs from the authors' ECBM (max diff {worst:.2e})"
    print("OK: port matches the authors' ECBM")


if __name__ == "__main__":
    main()
