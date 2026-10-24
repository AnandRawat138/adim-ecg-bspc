"""Aggregate the BSPC study runs into manuscript tables, statistics and figures.

Every number and plotted point comes from files written by run_bspc_study.py,
run_bspc_faithfulness.py and run_bspc_federated.py. Nothing is typed in by hand.

Usage:
    python -m experiments.analyze_bspc --runs experiments/runs/bspc --out ../BSPC_ADIM_Manuscript
Outputs:
    <out>/generated/results_summary.json     all aggregated numbers
    <out>/generated/tab_*.tex                 LaTeX table bodies (\\input in main.tex)
    <out>/figures/fig_*.pdf                   result figures (vector)
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
# Reference categorical palette (validated, light mode), fixed order
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

NAMES = {
    "it": "InceptionTime", "it_tr": "InceptionTime + Transformer",
    "it_tr_bl": "InceptionTime + Transformer + BiLSTM", "adim": "InceptionTime + Transformer + BiLSTM + ADIM",
    "se": "SE recalibration", "cbam": "CBAM", "tattn": "Temporal attention",
    "wo_lgig": "w/o LGIG (v1)", "wo_ain": "w/o AIN (v1)", "wo_ice": "w/o ICE (v1)", "wo_res": "w/o residual (v1)",
    "adim2": "InceptionTime + Transformer + BiLSTM + ADIM",
    "adim2_wo_lgig": "w/o LGIG", "adim2_wo_ain": "w/o AIN", "adim2_wo_ice": "w/o ICE",
    "adim2_wo_res": "w/o residual path",
}
NAMES["adim"] = "ADIM, original formulation (v1)"
MAIN = "adim2"  # proposed method

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load_runs(runs_dir):
    runs = defaultdict(dict)
    for p in glob.glob(os.path.join(runs_dir, "*", "seed*", "results.json")):
        cfg = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if cfg == "federated":
            continue
        r = json.load(open(p))
        r["_dir"] = os.path.dirname(p)
        runs[cfg][r["seed"]] = r
    return runs


def ms(vals, nd=4):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{vals.mean():.{nd}f} $\\pm$ {vals.std(ddof=1):.{nd}f}"


def stat(vals):
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v)),
            "values": [float(x) for x in v]}


def preds(run):
    z = np.load(os.path.join(run["_dir"], "test_preds.npz"))
    return z["ecg_id"], z["y"], z["prob"]


def macro_auc(y, p):
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in range(p.shape[1])]))


def paired_bootstrap(runs, a, b, seeds, B=2000, rng_seed=0):
    """Seed-averaged paired difference (a - b) on fold 10, with record-level bootstrap CI."""
    ya, pa, pb = None, [], []
    for s in seeds:
        ida, y1, p1 = preds(runs[a][s])
        idb, y2, p2 = preds(runs[b][s])
        assert np.array_equal(ida, idb) and np.array_equal(y1, y2)
        ya = y1
        pa.append(p1)
        pb.append(p2)
    n = len(ya)
    rng = np.random.default_rng(rng_seed)

    def diffs(idx):
        df, da = [], []
        for p1, p2 in zip(pa, pb):
            yy = ya[idx]
            df.append(f1_score(yy, p1[idx].argmax(1), average="macro", labels=range(5), zero_division=0)
                      - f1_score(yy, p2[idx].argmax(1), average="macro", labels=range(5), zero_division=0))
            da.append(macro_auc(yy, p1[idx]) - macro_auc(yy, p2[idx]))
        return np.mean(df), np.mean(da)

    obs = diffs(np.arange(n))
    boots = np.array([diffs(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"delta_f1": float(obs[0]), "ci_f1": [float(lo[0]), float(hi[0])],
            "delta_auc": float(obs[1]), "ci_auc": [float(lo[1]), float(hi[1])],
            "p_f1_le0": float(np.mean(boots[:, 0] <= 0)), "p_auc_le0": float(np.mean(boots[:, 1] <= 0)),
            "seeds": list(seeds), "B": B, "n_records": int(n)}


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "experiments/runs/bspc"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), "BSPC_ADIM_Manuscript"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    gen = os.path.join(args.out, "generated")
    figd = os.path.join(args.out, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figd, exist_ok=True)

    runs = load_runs(args.runs)
    S = {"configs": {}, "missing": []}
    for cfg in NAMES:
        if cfg not in runs:
            S["missing"].append(cfg)
            continue
        rs = [runs[cfg][s] for s in sorted(runs[cfg])]
        S["configs"][cfg] = {
            "seeds": sorted(runs[cfg]),
            "test_macro_f1": stat([r["test"]["macro_f1"] for r in rs]),
            "test_macro_auc": stat([r["test"]["macro_auc"] for r in rs]),
            "test_accuracy": stat([r["test"]["accuracy"] for r in rs]),
            "val_macro_f1": stat([r["val"]["macro_f1"] for r in rs]),
            "val_macro_auc": stat([r["val"]["macro_auc"] for r in rs]),
            "best_epoch": stat([r["best_epoch"] for r in rs]),
            "params": rs[0]["complexity"]["params"],
            "flops_M": rs[0]["complexity"]["flops_M"],
            "latency_ms": stat([r["complexity"]["latency_ms_bs1"] for r in rs]),
            "throughput_sps": stat([r["complexity"]["throughput_sps_bs128"] for r in rs]),
            "train_time_min": stat([r["train_time_s"] / 60 for r in rs]),
            "per_class": {c: {k: stat([r["test"]["per_class"][c][k] for r in rs])
                              for k in ("precision", "recall", "f1", "auc")} for c in CLASSES},
            "support": {c: rs[0]["test"]["per_class"][c]["support"] for c in CLASSES},
            "n_train_val_test": [rs[0]["n_train"], rs[0]["n_val"], rs[0]["n_test"]],
            "env": rs[0]["env"],
        }
    C = S["configs"]

    def row(cfg, cols):
        c = C[cfg]
        out = [NAMES[cfg]]
        for col in cols:
            if col == "params":
                out.append(f"{c['params']:,}".replace(",", "{,}"))
            elif col == "flops":
                out.append(f"{c['flops_M']:,.0f}".replace(",", "{,}"))
            elif col == "lat":
                out.append(f"{c['latency_ms']['mean']:.2f}")
            elif col == "f1":
                out.append(ms(c["test_macro_f1"]["values"]))
            elif col == "auc":
                out.append(ms(c["test_macro_auc"]["values"]))
            elif col == "acc":
                out.append(ms(c["test_accuracy"]["values"]))
            elif col == "vf1":
                out.append(ms(c["val_macro_f1"]["values"]))
        return " & ".join(out) + r" \\"

    # ---- Table: architecture progression
    arch = [k for k in ("it", "it_tr", "it_tr_bl", MAIN) if k in C]
    write(os.path.join(gen, "tab_architecture.tex"),
          "\n".join(row(k, ["params", "flops", "lat", "f1", "auc"]) for k in arch) + "\n")
    # ---- Table: refinement comparison
    ref = [k for k in ("it_tr_bl", "se", "cbam", "tattn", MAIN) if k in C]
    names_ref = {"it_tr_bl": "None (identity)", MAIN: "ADIM"}
    lines = []
    for k in ref:
        c = C[k]
        lines.append(" & ".join([names_ref.get(k, NAMES[k]), f"{c['params']:,}".replace(",", "{,}"),
                                 ms(c["test_macro_f1"]["values"]), ms(c["test_macro_auc"]["values"]),
                                 ms(c["test_accuracy"]["values"])]) + r" \\")
    write(os.path.join(gen, "tab_refinement.tex"), "\n".join(lines) + "\n")

    # ---- Table: ADIM component ablation (per-seed paired differences vs full ADIM)
    abl = [k for k in ("adim2_wo_lgig", "adim2_wo_ain", "adim2_wo_ice", "adim2_wo_res") if k in C]
    S["ablation"] = {}
    lines = []
    if MAIN in C:
        lines.append(" & ".join(["Full ADIM", ms(C[MAIN]["test_macro_f1"]["values"]),
                                 ms(C[MAIN]["test_macro_auc"]["values"]), "--", "--"]) + r" \\")
    for k in abl:
        common = sorted(set(runs[k]) & set(runs.get(MAIN, {})))
        if not common:
            continue
        dF = [runs[k][s]["test"]["macro_f1"] - runs[MAIN][s]["test"]["macro_f1"] for s in common]
        dA = [runs[k][s]["test"]["macro_auc"] - runs[MAIN][s]["test"]["macro_auc"] for s in common]
        S["ablation"][k] = {"seeds": common, "delta_f1": stat(dF), "delta_auc": stat(dA)}
        sgn = lambda v: ms(v).replace("$\\pm$", "$\\pm$") if len(v) > 1 else f"{v[0]:+.4f}"
        lines.append(" & ".join([NAMES[k], ms(C[k]["test_macro_f1"]["values"]), ms(C[k]["test_macro_auc"]["values"]),
                                 f"{np.mean(dF):+.4f}", f"{np.mean(dA):+.4f}"]) + r" \\")
    write(os.path.join(gen, "tab_ablation.tex"), "\n".join(lines) + "\n")

    # ---- Paired bootstrap: ADIM vs each comparator
    S["bootstrap"] = {}
    if MAIN in C:
        for k in ("it", "it_tr", "it_tr_bl", "se", "cbam", "tattn", "adim",
                  "adim2_wo_lgig", "adim2_wo_ain", "adim2_wo_ice", "adim2_wo_res"):
            if k in C:
                common = sorted(set(runs[k]) & set(runs[MAIN]))
                S["bootstrap"][k] = paired_bootstrap(runs, MAIN, k, common, B=args.bootstrap)
        lines = []
        for k, b in S["bootstrap"].items():
            if k.startswith("adim"):
                continue
            lines.append(" & ".join([NAMES[k].replace("InceptionTime", "IT").replace("Transformer", "Trans."),
                                     f"{b['delta_f1']:+.4f}", f"[{b['ci_f1'][0]:+.4f}, {b['ci_f1'][1]:+.4f}]",
                                     f"{b['delta_auc']:+.4f}", f"[{b['ci_auc'][0]:+.4f}, {b['ci_auc'][1]:+.4f}]"]) + r" \\")
        write(os.path.join(gen, "tab_bootstrap.tex"), "\n".join(lines) + "\n")

    # ---- Table: class-wise ADIM performance
    if MAIN in C:
        pc = C[MAIN]["per_class"]
        lines = [" & ".join([c, str(C[MAIN]["support"][c]), ms(pc[c]["precision"]["values"], 3),
                             ms(pc[c]["recall"]["values"], 3), ms(pc[c]["f1"]["values"], 3),
                             ms(pc[c]["auc"]["values"], 3)]) + r" \\" for c in CLASSES]
        write(os.path.join(gen, "tab_classwise.tex"), "\n".join(lines) + "\n")
        if "it_tr_bl" in C:
            S["classwise_delta_vs_it_tr_bl"] = {
                c: {k: C[MAIN]["per_class"][c][k]["mean"] - C["it_tr_bl"]["per_class"][c][k]["mean"]
                    for k in ("precision", "recall", "f1", "auc")} for c in CLASSES}

    # ---- Faithfulness (masking) and concentration
    F = defaultdict(list)
    for cfg in (MAIN, "tattn", "adim"):
        for p in sorted(glob.glob(os.path.join(args.runs, cfg, "seed*", "faithfulness.json"))):
            F[cfg].append(json.load(open(p)))
    S["faithfulness"] = {}
    for cfg, lst in F.items():
        agg = defaultdict(lambda: defaultdict(list))
        for r in lst:
            agg[("none", 0.0)]["macro_f1"].append(r["unmasked"]["macro_f1"])
            agg[("none", 0.0)]["macro_auc"].append(r["unmasked"]["macro_auc"])
            for m in r["masking"]:
                agg[(m["strategy"], m["fraction"])]["macro_f1"].append(m["macro_f1"])
                agg[(m["strategy"], m["fraction"])]["macro_auc"].append(m["macro_auc"])
        S["faithfulness"][cfg] = {f"{k[0]}@{k[1]:.1f}": {m: stat(v) for m, v in d.items()} for k, d in agg.items()}
        S["faithfulness"][cfg + "_concentration"] = {
            k: stat([r["concentration"][k] for r in lst])
            for k in ("mean", "sd", "auroc_correct_vs_incorrect_c", "auroc_correct_vs_incorrect_maxprob",
                      "spearman_c_maxprob")}
        S["faithfulness"][cfg + "_concentration"]["per_class_mean"] = {
            c: stat([r["concentration"]["per_class_mean"][c] for r in lst]) for c in CLASSES}
        S["faithfulness"][cfg + "_refinement_term"] = [r.get("refinement_term") for r in lst]
    if F:
        lines = []
        for frac in (0.1, 0.2, 0.3):
            cells = [f"{int(frac * 100)}\\%"]
            for cfg in (MAIN, "tattn"):
                for strat in ("high", "random", "low"):
                    v = S["faithfulness"].get(cfg, {}).get(f"{strat}@{frac:.1f}")
                    cells.append(ms(v["macro_f1"]["values"], 3) if v else "--")
            lines.append(" & ".join(cells) + r" \\")
        base = ["0\\%"] + [ms(S["faithfulness"][cfg]["none@0.0"]["macro_f1"]["values"], 3) if cfg in S["faithfulness"]
                           else "--" for cfg in (MAIN, "tattn") for _ in range(3)]
        write(os.path.join(gen, "tab_faithfulness.tex"), " & ".join(base) + r" \\" + "\n\\midrule\n" + "\n".join(lines) + "\n")

    # ---- Federated
    S["federated"] = {}
    for p in sorted(glob.glob(os.path.join(args.runs, "federated", "*", "seed*", "results.json"))):
        r = json.load(open(p))
        S["federated"][r["tag"]] = {k: r[k] for k in ("algorithm", "partition", "alpha", "mu", "clients",
                                                     "rounds_run", "best_round", "client_sizes",
                                                     "client_class_counts", "payload_bytes_per_round_up_and_down",
                                                     "wall_time_s")}
        S["federated"][r["tag"]]["test_macro_f1"] = r["test"]["macro_f1"]
        S["federated"][r["tag"]]["test_macro_auc"] = r["test"]["macro_auc"]
    if S["federated"] and MAIN in C:
        lines = [" & ".join(["Centralised (seeds " + ", ".join(map(str, C[MAIN]["seeds"])) + ")", "--", "--",
                             ms(C[MAIN]["test_macro_f1"]["values"]), ms(C[MAIN]["test_macro_auc"]["values"])]) + r" \\"]
        if 42 in runs[MAIN]:
            lines.append(" & ".join(["Centralised (seed 42)", "--", "--", f"{runs[MAIN][42]['test']['macro_f1']:.4f}",
                                     f"{runs[MAIN][42]['test']['macro_auc']:.4f}"]) + r" \\")
        lab = {"fedavg_iid_K5": ("FedAvg", "IID"), "fedavg_dirichlet0.5_K5": ("FedAvg", "Dirichlet $\\alpha=0.5$"),
               "fedprox_dirichlet0.5_K5": ("FedProx ($\\mu=0.01$)", "Dirichlet $\\alpha=0.5$")}
        for tag, r in S["federated"].items():
            a, p_ = lab.get(tag, (r["algorithm"], r["partition"]))
            lines.append(" & ".join([a, p_, str(r["best_round"]), f"{r['test_macro_f1']:.4f}",
                                     f"{r['test_macro_auc']:.4f}"]) + r" \\")
        write(os.path.join(gen, "tab_federated.tex"), "\n".join(lines) + "\n")

    json.dump(S, open(os.path.join(gen, "results_summary.json"), "w"), indent=1)

    # ================================================================ figures
    if MAIN in C:
        seeds = C[MAIN]["seeds"]
        # Confusion matrix: counts pooled over seeds, row-normalised
        cm = np.zeros((5, 5))
        for s in seeds:
            cm += np.array(runs[MAIN][s]["test"]["confusion_matrix"])
        cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(3.3, 2.9))
        ax.grid(False)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(5), CLASSES)
        ax.set_yticks(range(5), CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Reference class")
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)
        cb.set_label("Row-normalised proportion", fontsize=6.8)
        fig.savefig(os.path.join(figd, "fig_confusion_matrix.pdf"))
        fig.savefig(os.path.join(figd, "fig_confusion_matrix.png"), dpi=150)
        plt.close(fig)

        # ROC curves (seed with median macro-F1), legend reports mean ± SD AUC over seeds
        med = sorted(seeds, key=lambda s: runs[MAIN][s]["test"]["macro_f1"])[len(seeds) // 2]
        _, y, p = preds(runs[MAIN][med])
        fig, ax = plt.subplots(figsize=(3.3, 2.9))
        for c in range(5):
            fpr, tpr, _ = roc_curve((y == c).astype(int), p[:, c])
            a = C[MAIN]["per_class"][CLASSES[c]]["auc"]
            ax.plot(fpr, tpr, color=PAL[c], lw=1.3,
                    label=f"{CLASSES[c]}  AUC {a['mean']:.3f}" + (f" $\\pm$ {a['sd']:.3f}" if a["n"] > 1 else ""))
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.legend(loc="lower right", frameon=False, handlelength=1.4)
        fig.savefig(os.path.join(figd, "fig_roc_curves.pdf"))
        fig.savefig(os.path.join(figd, "fig_roc_curves.png"), dpi=150)
        plt.close(fig)
        S["roc_plot_seed"] = int(med)

        # Training curves: validation macro-F1 per epoch
        import csv
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=False)
        for k, col, nm in (("it_tr_bl", PAL[0], "without ADIM"), (MAIN, PAL[1], "with ADIM")):
            if k not in runs:
                continue
            for i, s in enumerate(sorted(runs[k])):
                h = list(csv.DictReader(open(os.path.join(runs[k][s]["_dir"], "history.csv"))))
                ep = [int(r["epoch"]) for r in h]
                axs[0].plot(ep, [float(r["val_f1"]) for r in h], color=col, lw=1.0, alpha=0.9,
                            label=nm if i == 0 else None)
                axs[1].plot(ep, [float(r["train_loss"]) for r in h], color=col, lw=1.0, alpha=0.9)
                axs[1].plot(ep, [float(r["val_loss"]) for r in h], color=col, lw=1.0, alpha=0.9, ls=(0, (3, 1.5)))
                be = runs[k][s]["best_epoch"]
                axs[0].plot([be], [runs[k][s]["best_val_macro_f1"]], "o", ms=3.5, color=col, mec="white", mew=0.6)
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Validation macro-F1 (fold 9)")
        axs[0].legend(frameon=False, loc="lower right")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].plot([], [], color=MUTED, lw=1.0, label="training")
        axs[1].plot([], [], color=MUTED, lw=1.0, ls=(0, (3, 1.5)), label="validation")
        axs[1].legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(figd, "fig_training_curves.pdf"))
        fig.savefig(os.path.join(figd, "fig_training_curves.png"), dpi=150)
        plt.close(fig)

    # Complexity vs macro-F1 (single series, direct labels)
    pts = [k for k in ("it", "it_tr", "it_tr_bl", "se", "cbam", "tattn", MAIN) if k in C]
    if pts:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        short = {"it": "IT", "it_tr": "IT+Tr", "it_tr_bl": "IT+Tr+BiLSTM", "se": "+SE", "cbam": "+CBAM",
                 "tattn": "+Temp. attn", MAIN: "+ADIM"}
        for k in pts:
            c = C[k]
            col = PAL[1] if k == MAIN else PAL[0]
            ax.errorbar(c["latency_ms"]["mean"], c["test_macro_f1"]["mean"], yerr=c["test_macro_f1"]["sd"],
                        fmt="o", ms=5, color=col, mec="white", mew=0.7, elinewidth=0.8, capsize=0)
            ax.annotate(short[k], (c["latency_ms"]["mean"], c["test_macro_f1"]["mean"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=6.4, color=INK)
        ax.set_xlabel("Latency, batch size 1 (ms)")
        ax.set_ylabel("Test macro-F1 (mean $\\pm$ SD)")
        fig.savefig(os.path.join(figd, "fig_complexity_tradeoff.pdf"))
        fig.savefig(os.path.join(figd, "fig_complexity_tradeoff.png"), dpi=150)
        plt.close(fig)

    # Ablation: paired differences vs full ADIM
    if S.get("ablation"):
        ks = list(S["ablation"])
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 1.9), sharey=True)
        for ax, met, ttl in ((axs[0], "delta_f1", "$\\Delta$ macro-F1 vs. full ADIM"),
                             (axs[1], "delta_auc", "$\\Delta$ macro-AUC vs. full ADIM")):
            for i, k in enumerate(ks):
                d = S["ablation"][k][met]
                ax.errorbar(d["mean"], i, xerr=d["sd"] if d["n"] > 1 else None, fmt="o", ms=5, color=PAL[0],
                            mec="white", mew=0.7, elinewidth=0.8)
                for v in d["values"]:
                    ax.plot(v, i, "|", color=MUTED, ms=6, mew=0.8)
            ax.axvline(0, color=MUTED, lw=0.8)
            ax.set_xlabel(ttl)
            ax.grid(axis="y", visible=False)
        axs[0].set_yticks(range(len(ks)), [NAMES[k] for k in ks])
        axs[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(figd, "fig_ablation_study.pdf"))
        fig.savefig(os.path.join(figd, "fig_ablation_study.png"), dpi=150)
        plt.close(fig)

    # Faithfulness curves
    if S["faithfulness"]:
        cf = [c for c in (MAIN, "tattn") if c in S["faithfulness"]]
        fig, axs = plt.subplots(1, len(cf), figsize=(3.4 * len(cf), 2.4), sharey=True, squeeze=False)
        for ax, cfg in zip(axs[0], cf):
            d = S["faithfulness"][cfg]
            for strat, col, nm in (("high", PAL[1], "highest-weight positions"), ("random", MUTED, "random positions"),
                                   ("low", PAL[0], "lowest-weight positions")):
                fr = [f for f in (0.1, 0.2, 0.3) if f"{strat}@{f:.1f}" in d]
                xs = [0] + [int(round(f * 100)) for f in fr]
                ys = [d["none@0.0"]["macro_f1"]["mean"]] + [d[f"{strat}@{f:.1f}"]["macro_f1"]["mean"] for f in fr]
                es = [d["none@0.0"]["macro_f1"]["sd"]] + [d[f"{strat}@{f:.1f}"]["macro_f1"]["sd"] for f in fr]
                ax.errorbar(xs, ys, yerr=es, color=col, marker="o", ms=4, mec="white", mew=0.6, lw=1.3,
                            elinewidth=0.7, label=nm)
            ax.set_title("ADIM inspection map" if cfg == MAIN else "Temporal-attention map", color=INK)
            ax.set_xlabel("Masked time positions (%)")
            ax.set_xticks([0, 10, 20, 30])
        axs[0][0].set_ylabel("Test macro-F1 (mean $\\pm$ SD)")
        axs[0][0].legend(frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(figd, "fig_faithfulness.pdf"))
        fig.savefig(os.path.join(figd, "fig_faithfulness.png"), dpi=150)
        plt.close(fig)

    # Inspection-map examples: for each class, the lowest ecg_id test record correctly classified by ADIM (seed 42)
    mp = os.path.join(args.runs, MAIN, "seed42", "test_maps.npz")
    if os.path.exists(mp):
        z = np.load(mp)
        ids, y, p, A = z["ecg_id"], z["y"], z["prob"], z["maps"].astype(np.float32)
        sig = np.load(os.path.join(ROOT, "data/cache/signals_6a0ac778591c3b709de7f0384af565b2.npy"), mmap_mode="r")
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data/ptbxl/ptbxl_database.csv"), index_col="ecg_id")
        pos = {e: i for i, e in enumerate(df.index)}
        chosen = []
        for c in range(5):
            cand = [i for i in np.argsort(ids) if y[i] == c and p[i].argmax() == c]
            if cand:
                chosen.append(cand[0])
        fig, axs = plt.subplots(len(chosen), 1, figsize=(6.8, 1.15 * len(chosen)), sharex=True)
        t = np.arange(A.shape[1]) / 100.0
        S["map_examples"] = []
        for ax, i in zip(np.atleast_1d(axs), chosen):
            x = sig[pos[int(ids[i])]][1]  # lead II
            ax.grid(False)
            ax.plot(t, x, color=INK, lw=0.6)
            ax.set_yticks([])
            ax2 = ax.twinx()
            ax2.fill_between(t, 0, A[i] * A.shape[1], color=PAL[1], alpha=0.35, lw=0)
            ax2.axhline(1.0, color=PAL[1], lw=0.5, ls=(0, (2, 2)))
            ax2.set_yticks([])
            ax2.set_ylim(0, max(2.0, float((A[i] * A.shape[1]).max()) * 1.05))
            ax2.spines[["top", "right", "left"]].set_visible(False)
            ax.spines[["left"]].set_visible(False)
            H = -(A[i] * np.log(A[i] + 1e-9)).sum() / np.log(A.shape[1])
            ax.text(0.003, 0.97, f"{CLASSES[y[i]]}  (ECG {int(ids[i])}, p={p[i].max():.2f}, c={1 - H:.3f})",
                    transform=ax.transAxes, va="top", fontsize=6.6, color=INK)
            ax2.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)
            S["map_examples"].append({"ecg_id": int(ids[i]), "class": CLASSES[y[i]], "prob": float(p[i].max()),
                                      "concentration": float(1 - H)})
        np.atleast_1d(axs)[-1].set_xlabel("Time (s)")
        fig.tight_layout(h_pad=0.3)
        fig.savefig(os.path.join(figd, "fig_inspection_maps.pdf"))
        fig.savefig(os.path.join(figd, "fig_inspection_maps.png"), dpi=150)
        plt.close(fig)
        json.dump(S, open(os.path.join(gen, "results_summary.json"), "w"), indent=1)

    print(json.dumps({k: {"f1": v["test_macro_f1"]["mean"], "auc": v["test_macro_auc"]["mean"],
                          "n": v["test_macro_f1"]["n"]} for k, v in C.items()}, indent=1))
    print("missing:", S["missing"])


if __name__ == "__main__":
    main()
