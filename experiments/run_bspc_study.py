"""Single-label five-class PTB-XL study used for the BSPC manuscript.

Trains one (configuration, seed) pair on folds 1-8, selects the checkpoint with the best
validation (fold 9) macro-F1, and evaluates it once on fold 10. All outputs are written to
experiments/runs/bspc/<config>/seed<seed>/ :
    history.csv        per-epoch train/val loss, val macro-F1, LR
    weights/best_model.pt
    results.json       test metrics, complexity profile, run metadata
    test_preds.npz     ecg_id, integer targets, softmax probabilities (fold 10)
    val_preds.npz      same for fold 9

Usage:
    python -m experiments.run_bspc_study --config adim --seed 42
"""
import os
import sys
import json
import time
import argparse
import logging
import platform

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import models  # noqa: F401  (populates the registry)
from models.registry import ModelRegistry
from data.dataset import PTBXLDataset
from training.trainer import Trainer
from training.loss import get_loss_fn
from evaluation.evaluator import Evaluator
from utils.reproducibility import set_seed

logger = logging.getLogger("ecg_classification")

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Model switches for every configuration in the study
CONFIGS = {
    # Architecture progression
    "it":        dict(use_transformer=False, use_bilstm=False, use_adim=False, pooling="avg"),
    "it_tr":     dict(use_transformer=True,  use_bilstm=False, use_adim=False, pooling="avg"),
    "it_tr_bl":  dict(use_transformer=True,  use_bilstm=True,  use_adim=False, pooling="attention"),
    "adim":      dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention"),
    # Conventional refinement at the ADIM position
    "se":        dict(use_transformer=True,  use_bilstm=True,  use_adim=False, pooling="attention", refinement="se"),
    "cbam":      dict(use_transformer=True,  use_bilstm=True,  use_adim=False, pooling="attention", refinement="cbam"),
    "tattn":     dict(use_transformer=True,  use_bilstm=True,  use_adim=False, pooling="attention", refinement="tattn"),
    # ADIM component ablations
    "wo_lgig":   dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", disable_lgig=True),
    "wo_ain":    dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", disable_ain=True),
    "wo_ice":    dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", disable_certainty=True),
    "wo_res":    dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", disable_residual=True),
    # Revised ADIM (v2): temporal weight applied after normalisation, w = 1 + gamma*(1+c)*(T*a - 1)
    "adim2":         dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", adim_formulation="v2"),
    "adim2_wo_lgig": dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", adim_formulation="v2", disable_lgig=True),
    "adim2_wo_ain":  dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", adim_formulation="v2", disable_ain=True),
    "adim2_wo_ice":  dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", adim_formulation="v2", disable_certainty=True),
    "adim2_wo_res":  dict(use_transformer=True,  use_bilstm=True,  use_adim=True,  pooling="attention", adim_formulation="v2", disable_residual=True),
}


def seed_worker(worker_id):
    # Each DataLoader worker gets a distinct, seed-derived NumPy/random state for augmentation
    import random
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    random.seed(ws)


def load_cfg():
    cfg = OmegaConf.create({
        "env": OmegaConf.load(os.path.join(ROOT, "config/env/default.yaml")),
        "dataset": OmegaConf.load(os.path.join(ROOT, "config/dataset/default.yaml")),
        "training": OmegaConf.load(os.path.join(ROOT, "config/training/default.yaml")),
        "augmentation": OmegaConf.load(os.path.join(ROOT, "config/augmentation/default.yaml")),
        "model": OmegaConf.load(os.path.join(ROOT, "config/model/inception_time_hybrid.yaml")),
    })
    # Resolve the hydra-relative dataset paths without hydra
    cfg.dataset.raw_dir = os.path.join(ROOT, "data/ptbxl")
    cfg.dataset.cache_dir = os.path.join(ROOT, "data/cache")
    cfg.training.checkpoint.save_dir = os.path.join(ROOT, "checkpoints")
    cfg.training.output_dir = os.path.join(ROOT, "experiments/runs")
    cfg.training.loss.name = "softmax_focal"
    return cfg


def build_datasets(cfg):
    common = dict(raw_dir=cfg.dataset.raw_dir, cache_dir=cfg.dataset.cache_dir,
                  sampling_rate=cfg.dataset.sampling_rate, signal_length=cfg.dataset.signal_length,
                  classes=list(cfg.dataset.classes), use_caching=cfg.dataset.use_caching, label_mode="single")
    train = PTBXLDataset(split="train", augment=True,
                         aug_config=OmegaConf.to_container(cfg.augmentation, resolve=True), **common)
    val = PTBXLDataset(split="val", augment=False, **common)
    test = PTBXLDataset(split="test", augment=False, **common)
    return train, val, test


def build_model(cfg, config_name):
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name")
    model_cfg.update(CONFIGS[config_name])
    return ModelRegistry.create("inception_time_hybrid", **model_cfg)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        with torch.amp.autocast(device_type=device.type):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def single_label_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    aucs = [roc_auc_score((y_true == c).astype(int), y_prob[:, c]) for c in labels]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_auc": float(np.mean(aucs)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "auc": float(aucs[i]), "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="override (debug only)")
    ap.add_argument("--out_root", default=os.path.join(ROOT, "experiments/runs/bspc"))
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers for training augmentation")
    ap.add_argument("--deterministic", type=int, default=0, help="1 = cuDNN deterministic mode (slower)")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.config, f"seed{args.seed}")
    if os.path.exists(os.path.join(run_dir, "results.json")):
        print(f"[skip] {run_dir} already complete")
        return
    for stale in ("history.csv", "train.log"):  # restart an interrupted run cleanly
        if os.path.exists(os.path.join(run_dir, stale)):
            os.remove(os.path.join(run_dir, stale))
    os.makedirs(run_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
                        handlers=[logging.FileHandler(os.path.join(run_dir, "train.log")), logging.StreamHandler()])
    cfg = load_cfg()
    if args.epochs:
        cfg.training.epochs = args.epochs
    cfg.env.seed = args.seed
    cfg.env.deterministic = bool(args.deterministic)
    logger.info(f"Config={args.config} seed={args.seed} switches={CONFIGS[args.config]}")

    set_seed(args.seed, deterministic=cfg.env.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = build_datasets(cfg)
    g = torch.Generator()
    g.manual_seed(args.seed)
    nw = args.workers
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=nw, generator=g,
                              worker_init_fn=seed_worker, persistent_workers=nw > 0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg, args.config).to(device)
    hw = Evaluator(model, device, CLASSES).profile_hardware(
        input_shape=(1, cfg.dataset.num_leads, cfg.dataset.signal_length))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr,
                                  weight_decay=cfg.training.optimizer.weight_decay)
    loss_fn = get_loss_fn(cfg.training.loss.name, label_smoothing=cfg.training.loss.label_smoothing,
                          focal_alpha=cfg.training.loss.focal_alpha, focal_gamma=cfg.training.loss.focal_gamma)

    t0 = time.time()
    trainer = Trainer(model=model, cfg=cfg, train_loader=train_loader, val_loader=val_loader,
                      optimizer=optimizer, loss_fn=loss_fn, device=device, run_dir=run_dir)
    trainer.fit()
    train_time = time.time() - t0

    ckpt = torch.load(os.path.join(run_dir, "weights/best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = int(ckpt["epoch"])

    val_prob, val_true = predict(model, val_loader, device)
    test_prob, test_true = predict(model, test_loader, device)
    np.savez_compressed(os.path.join(run_dir, "val_preds.npz"), ecg_id=val_ds.indices, y=val_true, prob=val_prob)
    np.savez_compressed(os.path.join(run_dir, "test_preds.npz"), ecg_id=test_ds.indices, y=test_true, prob=test_prob)

    results = {
        "config": args.config, "seed": args.seed, "switches": CONFIGS[args.config],
        "best_epoch": best_epoch, "best_val_macro_f1": float(ckpt["best_metric"]),
        "train_time_s": train_time,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "val": single_label_metrics(val_true, val_prob),
        "test": single_label_metrics(test_true, test_prob),
        "complexity": {"params": int(hw["total_parameters"]), "flops_M": hw["theoretical_flops"] / 1e6,
                       "latency_ms_bs1": hw["inference_latency_ms"],
                       "throughput_sps_bs128": hw["throughput_samples_per_sec"]},
        "loader": {"train_workers": args.workers, "deterministic_cudnn": bool(args.deterministic)},
        "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "training_cfg": OmegaConf.to_container(cfg.training, resolve=True),
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"DONE {args.config} seed{args.seed}: test macro-F1={results['test']['macro_f1']:.4f} "
                f"macro-AUC={results['test']['macro_auc']:.4f} (best epoch {best_epoch})")


if __name__ == "__main__":
    main()
