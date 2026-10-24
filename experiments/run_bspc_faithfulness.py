"""Perturbation (masking) analysis of learned temporal weights, plus inspection-concentration analysis.

For a trained model with a temporal weighting map (ADIM inspection map or the temporal-attention
baseline), each fold-10 record is passed through once to obtain its map a in R^T (T = 1000, the
same length as the 100 Hz input). The highest-, lowest- or randomly-ranked k% of time positions
are then replaced, in all 12 leads, by that record's per-lead temporal mean, and the masked record
is classified again. Random masking is repeated with --random_repeats draws and averaged.

The analysis measures how strongly predictions depend on the time positions the map emphasises.
It does not measure clinical correctness.

It also records the per-record inspection concentration c = 1 - H(a)/log T and tests how well it
separates correct from incorrect predictions (AUROC), alongside the maximum softmax probability.

Usage:
    python -m experiments.run_bspc_faithfulness --config adim --seed 42
"""
import os
import sys
import json
import math
import argparse

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_bspc_study import load_cfg, build_datasets, build_model, single_label_metrics
from utils.reproducibility import set_seed


def map_module(model, config):
    return model.adim if config.startswith("adim") or config.startswith("wo_") else model.refine


@torch.no_grad()
def collect(model, loader, device, mod):
    xs, ys, probs, maps = [], [], [], []
    for x, y in loader:
        with torch.amp.autocast(device_type=device.type):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        maps.append(mod.last_attention_map.float().cpu().numpy())
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(probs), np.concatenate(maps)


@torch.no_grad()
def predict_array(model, X, device, bs=128):
    out = []
    for i in range(0, len(X), bs):
        with torch.amp.autocast(device_type=device.type):
            logits = model(torch.from_numpy(X[i:i + bs]).to(device))
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def stat_list(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def mask(X, order, k):
    Xm = X.copy()
    fill = X.mean(axis=2, keepdims=True)  # per-record, per-lead temporal mean
    for i in range(len(X)):
        pos = order[i, :k]
        Xm[i][:, pos] = fill[i][:, 0][:, None]
    return Xm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="adim")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fractions", default="0.1,0.2,0.3")
    ap.add_argument("--random_repeats", type=int, default=3)
    ap.add_argument("--runs", default=os.path.join(ROOT, "experiments/runs/bspc"))
    args = ap.parse_args()

    run_dir = os.path.join(args.runs, args.config, f"seed{args.seed}")
    out_path = os.path.join(run_dir, "faithfulness.json")
    if os.path.exists(out_path):
        print(f"[skip] {out_path}")
        return
    cfg = load_cfg()
    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_ds = build_datasets(cfg)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = build_model(cfg, args.config).to(device)
    ckpt = torch.load(os.path.join(run_dir, "weights/best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mod = map_module(model, args.config)

    X, y, p0, A = collect(model, loader, device, mod)
    T = A.shape[1]
    assert T == X.shape[2], "temporal map must align with input samples"
    np.savez_compressed(os.path.join(run_dir, "test_maps.npz"), ecg_id=test_ds.indices, y=y, prob=p0,
                        maps=A.astype(np.float16))

    res = {"config": args.config, "seed": args.seed, "T": T, "unmasked": single_label_metrics(y, p0), "masking": []}
    order_high = np.argsort(-A, axis=1)
    order_low = np.argsort(A, axis=1)
    rng = np.random.default_rng(args.seed)
    for frac in [float(f) for f in args.fractions.split(",")]:
        k = int(round(frac * T))
        for name, order in [("high", order_high), ("low", order_low)]:
            m = single_label_metrics(y, predict_array(model, mask(X, order, k), device))
            res["masking"].append({"strategy": name, "fraction": frac, "macro_f1": m["macro_f1"],
                                   "macro_auc": m["macro_auc"], "accuracy": m["accuracy"]})
        rf1, rauc, racc = [], [], []
        for _ in range(args.random_repeats):
            order_r = np.argsort(rng.random(A.shape), axis=1)
            m = single_label_metrics(y, predict_array(model, mask(X, order_r, k), device))
            rf1.append(m["macro_f1"]); rauc.append(m["macro_auc"]); racc.append(m["accuracy"])
        res["masking"].append({"strategy": "random", "fraction": frac, "macro_f1": float(np.mean(rf1)),
                               "macro_f1_sd": float(np.std(rf1)), "macro_auc": float(np.mean(rauc)),
                               "macro_auc_sd": float(np.std(rauc)), "accuracy": float(np.mean(racc)),
                               "repeats": args.random_repeats})
        print(frac, [r for r in res["masking"] if r["fraction"] == frac])

    # Inspection concentration vs. predictive confidence
    H = -(A * np.log(A + 1e-9)).sum(axis=1) / math.log(T)
    c = 1.0 - H
    correct = (p0.argmax(1) == y).astype(int)
    maxp = p0.max(1)
    res["concentration"] = {
        "mean": float(c.mean()), "sd": float(c.std()), "min": float(c.min()), "max": float(c.max()),
        "per_class_mean": {cls: float(c[y == i].mean()) for i, cls in enumerate(test_ds.classes)},
        "auroc_correct_vs_incorrect_c": float(roc_auc_score(correct, c)),
        "auroc_correct_vs_incorrect_maxprob": float(roc_auc_score(correct, maxp)),
        "spearman_c_maxprob": float(spearmanr(c, maxp).correlation),
        "uniform_reference_note": "c = 0 for a uniform map; c = 1 for a one-hot map",
        "max_weight_times_T": stat_list(A.max(axis=1) * T),
    }
    # Magnitude of the concentration-guided refinement term (ADIM only)
    if hasattr(mod, "gamma") and getattr(mod, "formulation", "v1") == "v2":
        g = float(mod.gamma.detach().cpu())
        W = 1.0 + g * (1.0 + c[:, None]) * (T * A - 1.0)  # realised temporal weights w_t
        res["refinement_term"] = {"formulation": "v2", "gamma": g,
                                  "w_min": stat_list(W.min(axis=1)), "w_max": stat_list(W.max(axis=1)),
                                  "w_sd_over_t": stat_list(W.std(axis=1)),
                                  "note": "H_out = H + w_t * (W_p LN(H_c) + b); w_t = 1 everywhere for a uniform map"}
    elif hasattr(mod, "gamma"):
        g = float(mod.gamma.detach().cpu())
        term_max = g * c * A.max(axis=1)
        res["refinement_term"] = {"formulation": "v1", "gamma": g, "max_t_gamma_c_a": stat_list(term_max),
                                  "note": "H_ref = H_c * (1 + gamma*c*a_t); values near 0 mean H_ref ~= H_c"}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("concentration", "refinement_term") if k in res}, indent=2))


if __name__ == "__main__":
    main()
