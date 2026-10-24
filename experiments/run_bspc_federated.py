"""Simulated federated training of the ADIM model (secondary experiment of the BSPC study).

The single-label training folds (1-8) are split across K virtual clients at the patient level,
so that no patient appears on two clients. Two partitions are supported:
    iid        patients assigned uniformly at random
    dirichlet  label-skewed: for each class, the patients whose first record has that class are
               divided among clients with proportions drawn from Dirichlet(alpha)
Each round, every client trains for `local_epochs` epochs from the current global weights, and
the server averages the weights (and BatchNorm buffers) by client sample count (FedAvg). With
--mu > 0 a FedProx proximal term (mu/2)||w - w_global||^2 is added to the local loss.
The global model is checked on fold 9 after every round. The best round on fold 9 is evaluated
once on fold 10. This is a simulation on one machine: no networking, secure aggregation or
privacy mechanism is involved.

Usage:
    python -m experiments.run_bspc_federated --partition iid --clients 5 --seed 42
    python -m experiments.run_bspc_federated --partition dirichlet --alpha 0.5 --mu 0.01
"""
import os
import sys
import json
import copy
import time
import argparse
import logging

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_bspc_study import load_cfg, build_datasets, build_model, predict, single_label_metrics
from training.loss import get_loss_fn
from training.scheduler import get_scheduler
from utils.reproducibility import set_seed

logger = logging.getLogger("ecg_classification")


def partition_patients(train_ds, n_clients, scheme, alpha, rng):
    """Returns a list of index arrays (positions in train_ds) for each client."""
    df = train_ds.split_df
    labels = df[train_ds.classes].values.argmax(axis=1)
    pos = pd.DataFrame({"patient": df["patient_id"].values, "label": labels, "pos": np.arange(len(df))})
    first = pos.groupby("patient")["label"].first()
    patients = first.index.values
    owner = {}
    if scheme == "iid":
        perm = rng.permutation(patients)
        for i, p in enumerate(perm):
            owner[p] = i % n_clients
    elif scheme == "dirichlet":
        for c in np.unique(first.values):
            pc = rng.permutation(patients[first.values == c])
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(pc)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(pc, cuts)):
                for p in chunk:
                    owner[p] = k
    else:
        raise ValueError(scheme)
    pos["client"] = pos["patient"].map(owner)
    return [pos.loc[pos.client == k, "pos"].values for k in range(n_clients)], pos


def local_train(model, global_state, loader, cfg, lr, loss_fn, device, mu, scaler):
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.training.optimizer.weight_decay)
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.env.mixed_precision):
            loss = loss_fn(model(x), y)
        if mu > 0:
            prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
            loss = loss + 0.5 * mu * prox
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
        scaler.step(opt)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return copy.deepcopy(model.state_dict()), total / max(n, 1)


def fedavg(states, weights):
    w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    out = {}
    for k in states[0]:
        if states[0][k].dtype in (torch.int64, torch.long):  # e.g. BatchNorm num_batches_tracked
            out[k] = states[0][k].clone()
        else:
            out[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, states)).to(states[0][k].dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.0, help="FedProx coefficient (0 = FedAvg)")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    algo = "fedprox" if args.mu > 0 else "fedavg"
    tag = f"{algo}_{args.partition}" + (f"{args.alpha}" if args.partition == "dirichlet" else "") + f"_K{args.clients}"
    run_dir = os.path.join(ROOT, "experiments/runs/bspc/federated", tag, f"seed{args.seed}")
    if os.path.exists(os.path.join(run_dir, "results.json")):
        print(f"[skip] {run_dir}")
        return
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] - %(message)s",
                        handlers=[logging.FileHandler(os.path.join(run_dir, "train.log")), logging.StreamHandler()])

    cfg = load_cfg()
    set_seed(args.seed, deterministic=cfg.env.deterministic)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = build_datasets(cfg)
    client_idx, assign = partition_patients(train_ds, args.clients, args.partition, args.alpha, rng)
    sizes = [len(ix) for ix in client_idx]
    labels = assign["label"].values
    client_dist = [np.bincount(labels[ix], minlength=5).tolist() for ix in client_idx]
    logger.info(f"{tag}: client sizes {sizes}; class counts per client {client_dist}")

    loaders = []
    for k, ix in enumerate(client_idx):
        g = torch.Generator()
        g.manual_seed(args.seed * 100 + k)
        loaders.append(DataLoader(Subset(train_ds, ix), batch_size=cfg.training.batch_size,
                                  shuffle=True, num_workers=0, generator=g))
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg, "adim2").to(device)
    loss_fn = get_loss_fn(cfg.training.loss.name, label_smoothing=cfg.training.loss.label_smoothing,
                          focal_alpha=cfg.training.loss.focal_alpha, focal_gamma=cfg.training.loss.focal_gamma)
    # Same warm-up + cosine learning-rate profile as centralised training, indexed by round
    dummy_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=cfg.training.optimizer.lr)
    sched = get_scheduler(dummy_opt, cfg.training.scheduler, args.rounds, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.env.mixed_precision)

    n_params = sum(p.numel() for p in model.parameters())
    bytes_per_round = 2 * args.clients * sum(v.numel() * v.element_size() for v in model.state_dict().values())

    global_state = copy.deepcopy(model.state_dict())
    best_f1, best_round, wait, history = -1.0, 0, 0, []
    t0 = time.time()
    for r in range(1, args.rounds + 1):
        lr = dummy_opt.param_groups[0]["lr"]
        states, losses = [], []
        for k in range(args.clients):
            for _ in range(args.local_epochs):
                st, l = local_train(model, global_state, loaders[k], cfg, lr, loss_fn, device, args.mu, scaler)
            states.append(st)
            losses.append(l)
        global_state = fedavg(states, sizes)
        model.load_state_dict(global_state)
        vp, vy = predict(model, val_loader, device)
        vm = single_label_metrics(vy, vp)
        history.append({"round": r, "lr": lr, "client_loss": float(np.average(losses, weights=sizes)),
                        "val_f1": vm["macro_f1"], "val_auc": vm["macro_auc"]})
        logger.info(f"Round {r:02d} | lr {lr:.6f} | client loss {history[-1]['client_loss']:.4f} | "
                    f"val F1 {vm['macro_f1']:.4f} | val AUC {vm['macro_auc']:.4f}")
        sched.step()
        if vm["macro_f1"] > best_f1:
            best_f1, best_round, wait = vm["macro_f1"], r, 0
            torch.save(global_state, os.path.join(run_dir, "best_global.pt"))
        else:
            wait += 1
            if wait >= args.patience:
                logger.info(f"Early stopping after round {r}")
                break
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_global.pt"), map_location=device))
    tp, ty = predict(model, test_loader, device)
    np.savez_compressed(os.path.join(run_dir, "test_preds.npz"), ecg_id=test_ds.indices, y=ty, prob=tp)
    res = {"tag": tag, "algorithm": algo, "partition": args.partition, "alpha": args.alpha, "mu": args.mu,
           "clients": args.clients, "local_epochs": args.local_epochs, "rounds_run": len(history),
           "best_round": best_round, "best_val_macro_f1": best_f1, "client_sizes": sizes,
           "client_class_counts": client_dist, "params": n_params,
           "payload_bytes_per_round_up_and_down": int(bytes_per_round),
           "wall_time_s": time.time() - t0, "test": single_label_metrics(ty, tp)}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"DONE {tag}: test macro-F1 {res['test']['macro_f1']:.4f} AUC {res['test']['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
