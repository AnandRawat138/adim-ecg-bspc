# Concentration-guided temporal refinement (ADIM) for 12-lead ECG classification

This repository contains the code, configuration and **complete experiment records** for:

> V. Srivastava, A. Rawat, N. Gupta, R. Singh. *Concentration-Guided Temporal Refinement for 12-Lead ECG Classification: Formulation Analysis and Controlled Evaluation on PTB-XL.* Submitted to *Biomedical Signal Processing and Control*.

Every number, table and figure in the article is produced by the scripts here from the records in `experiments/runs/bspc/`.

## Task and protocol
- **Data:** PTB-XL v1.0.3, 100 Hz, 12 leads, 10 s. **Single-label five-class** subset: recordings whose diagnostic statements map to exactly one of NORM, MI, STTC, CD or HYP, 16,244 in total.
- **Splits:** official patient-disjoint folds: 1–8 train (12,957), 9 validation (1,637), 10 test (1,650).
- **Model:** InceptionTime (2 × 3 modules) → Transformer (2 layers, d = 128) → BiLSTM (2 × 128) → refinement block → attention pooling → softmax.
- **Training:** softmax focal loss (α = 0.25, γ = 2) with label smoothing 0.1; AdamW (lr 1e-3, weight decay 1e-4); 5-epoch warm-up, then cosine decay; batch size 128; ≤ 50 epochs; early stopping (patience 10) on validation macro-F1.
- **Evaluation:** fold 10 is evaluated **once** per trained model. Seeds are 42, 43 and 44.

## Layout
| Path | Content |
|---|---|
| `layers/adim.py` | ADIM. `formulation="v2"` is the evaluated module; `"v1"` is the original pre-normalisation form, which is inert (see the article, Sec. 3.3) |
| `layers/refinement.py` | Temporal attention, SE and CBAM blocks at the ADIM position |
| `models/inception_time_hybrid.py` | Backbone and switches |
| `data/` | PTB-XL loading, single-label mode, augmentation |
| `experiments/run_bspc_study.py` | Trains one configuration and seed; writes `results.json`, `history.csv`, predictions and the best checkpoint |
| `experiments/run_bspc_faithfulness.py` | Inspection maps, deletion (masking) analysis, concentration statistics |
| `experiments/run_bspc_federated.py` | FedAvg / FedProx simulation with patient-level IID or Dirichlet clients |
| `experiments/analyze_bspc.py` | Aggregates runs into tables, paired bootstrap intervals and figures |
| `experiments/runs/bspc/` | **All run records** (see below) |
| `tools/download_ptbxl.sh` | Fetches PTB-XL v1.0.3 from PhysioNet |

## Experiment records (`experiments/runs/bspc/`)
For every `<config>/seed<N>/`:
- `results.json`: test and validation metrics, per-class metrics, confusion matrix, complexity, environment;
- `history.csv` and `train.log`: per-epoch losses and validation macro-F1;
- `test_preds.npz` and `val_preds.npz`: ECG ids, labels and softmax probabilities;
- `weights/best_model.pt` (Git LFS): the selected checkpoint.

Where the masking analysis was run, `faithfulness.json` and `test_maps.npz` hold its results and the inspection maps. Federated runs are under `federated/`, and `queue.log` records the start and end of every run.

| Config | Description | Seeds run |
|---|---|---|
| `it_tr_bl` | no refinement (identity) | 42, 43, 44 |
| `adim2` | ADIM (v2), evaluated method | 42, 43, 44 |
| `tattn` | temporal attention | 42 |
| `adim` | ADIM v1 (original formulation), reported only in the audit | 42 |
| `federated/fedavg_iid_K5` | FedAvg, 5 IID patient-disjoint clients, ≤ 30 rounds | 42 |

SE, CBAM, the ADIM component ablations (`adim2_wo_*`), InceptionTime and IT + Transformer are implemented and scripted (`experiments/run_bspc_all2.sh`) but were **not run** for the article.

## Reproduce
```bash
pip install -r requirements.txt
bash tools/download_ptbxl.sh
python -m experiments.run_bspc_study --config adim2 --seed 42
python -m experiments.run_bspc_faithfulness --config adim2 --seed 42
python -m experiments.run_bspc_federated --partition iid --clients 5 --seed 42 --rounds 30 --patience 8
python -m experiments.analyze_bspc --runs experiments/runs/bspc --out paper_outputs
```
The first run builds a memory-mapped cache of the 100 Hz signals in `data/cache/`. Runs use cuDNN autotuning and are not bit-wise deterministic.

## Data licence
PTB-XL is distributed by PhysioNet under CC BY 4.0. It is **not** redistributed here; use `tools/download_ptbxl.sh`. Please cite Wagner et al., *Scientific Data* 7:154 (2020), and the PhysioNet record https://doi.org/10.13026/kfzx-aw45.
