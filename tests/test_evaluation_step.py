import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.registry import ModelRegistry
from evaluation.metrics import calculate_ece, compute_all_metrics
from evaluation.evaluator import Evaluator, estimate_flops

# Force registration
import models

def test_ece_calculation():
    """Verify that ECE computes correct bounds."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.85, 0.15, 0.95, 0.05])
    
    ece = calculate_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    # Since predictions align perfectly with targets, ECE should be very low
    assert ece < 0.15


def test_compute_all_metrics():
    """Verify that compute_all_metrics returns all required statistical and clinical metrics."""
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.75, 0.7], [0.1, 0.15]])
    classes = ["ClassA", "ClassB"]
    
    metrics = compute_all_metrics(y_true, y_prob, classes)
    
    # Check overall macro metrics existence
    overall = metrics["overall"]
    assert "macro_accuracy" in overall
    assert "macro_f1" in overall
    assert "macro_auc" in overall
    assert "macro_mcc" in overall
    assert "macro_cohen_kappa" in overall
    assert "macro_balanced_accuracy" in overall
    assert "macro_ece" in overall
    assert "macro_brier" in overall
    
    # Check per class dict
    per_class = metrics["per_class"]
    assert "ClassA" in per_class
    assert per_class["ClassA"]["mcc"] >= -1.0
    assert per_class["ClassA"]["cohen_kappa"] >= -1.0
    assert per_class["ClassA"]["brier_score"] >= 0.0


def test_evaluator_profiling():
    """Verify that the Evaluator executes hardware performance checks successfully."""
    device = torch.device("cpu")
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    
    model = ModelRegistry.create(
        "cnn",
        input_channels=12,
        num_classes=5,
        filters=[16],
        fc_dim=16,
        pooling="avg"
    ).to(device)
    
    evaluator = Evaluator(model, device, classes)
    profile = evaluator.profile_hardware(input_shape=(1, 12, 1000))
    
    assert "total_parameters" in profile
    assert "inference_latency_ms" in profile
    assert "throughput_samples_per_sec" in profile
    assert "theoretical_flops" in profile
    
    assert profile["total_parameters"] > 0
    assert profile["theoretical_flops"] > 0
    assert profile["inference_latency_ms"] > 0.0
