import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss
)
from typing import Dict, List, Tuple

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) for binary probabilities.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of confidence intervals to bin.

    Returns:
        float: Expected calibration error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Mask for current bin range
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += (bin_size / n_samples) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    threshold: float = 0.5
) -> Dict[str, any]:
    """Calculates publication-ready clinical and statistical metrics for multi-label predictions.

    Args:
        y_true: True multi-label binary indicators of shape (N, C).
        y_prob: Predicted class probabilities of shape (N, C).
        classes: List of string names for classes of length C.
        threshold: Decision threshold to binarize predictions.

    Returns:
        Dict: Parsed overall averages and per-class breakdowns.
    """
    y_pred = (y_prob > threshold).astype(float)
    num_classes = len(classes)
    
    per_class_metrics = {}
    
    # Statistical and Clinical Metric Lists
    accs, precs, recs, f1s, aucs, pr_aucs = [], [], [], [], [], []
    mccs, kappas, balanced_accs, eces, briers = [], [], [], [], []

    for i, class_name in enumerate(classes):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        y_pr = y_prob[:, i]
        
        # Base stats
        class_acc = accuracy_score(yt, yp)
        class_prec = precision_score(yt, yp, zero_division=0)
        class_rec = recall_score(yt, yp, zero_division=0)
        class_f1 = f1_score(yt, yp, zero_division=0)
        
        # AUCs
        try:
            class_auc = roc_auc_score(yt, y_pr)
        except ValueError:
            class_auc = 0.5  # Fallback if only one class exists in split
            
        p, r, _ = precision_recall_curve(yt, y_pr)
        class_pr_auc = auc(r, p)
        
        # Advanced medical metrics
        class_mcc = matthews_corrcoef(yt, yp)
        class_kappa = cohen_kappa_score(yt, yp)
        class_balanced = balanced_accuracy_score(yt, yp)
        class_ece = calculate_ece(yt, y_pr)
        class_brier = brier_score_loss(yt, y_pr)

        # Record per-class
        per_class_metrics[class_name] = {
            "accuracy": float(class_acc),
            "precision": float(class_prec),
            "recall": float(class_rec),
            "f1": float(class_f1),
            "auc": float(class_auc),
            "pr_auc": float(class_pr_auc),
            "mcc": float(class_mcc),
            "cohen_kappa": float(class_kappa),
            "balanced_accuracy": float(class_balanced),
            "ece": float(class_ece),
            "brier_score": float(class_brier)
        }
        
        accs.append(class_acc)
        precs.append(class_prec)
        recs.append(class_rec)
        f1s.append(class_f1)
        aucs.append(class_auc)
        pr_aucs.append(class_pr_auc)
        mccs.append(class_mcc)
        kappas.append(class_kappa)
        balanced_accs.append(class_balanced)
        eces.append(class_ece)
        briers.append(class_brier)

    # Compute overall macro averages
    macro_metrics = {
        "macro_accuracy": float(np.mean(accs)),
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "macro_f1": float(np.mean(f1s)),
        "macro_auc": float(np.mean(aucs)),
        "macro_pr_auc": float(np.mean(pr_aucs)),
        "macro_mcc": float(np.mean(mccs)),
        "macro_cohen_kappa": float(np.mean(kappas)),
        "macro_balanced_accuracy": float(np.mean(balanced_accs)),
        "macro_ece": float(np.mean(eces)),
        "macro_brier": float(np.mean(briers))
    }
    
    return {
        "overall": macro_metrics,
        "per_class": per_class_metrics
    }
