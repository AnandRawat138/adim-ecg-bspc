import os
import pytest
import torch
import numpy as np
from data.dataset import PTBXLDataset
from data.transforms import standardize, pad_or_crop, resample_signal

RAW_DIR = "C:/Users/anand/.gemini/antigravity/scratch/ecg_classification/data/ptbxl"
CACHE_DIR = "C:/Users/anand/.gemini/antigravity/scratch/ecg_classification/data/cache_test"

def test_transforms():
    """Verify that signal transformations behave as expected."""
    # Dummy signal of shape (12, 1000)
    signal = np.random.randn(12, 1000)
    
    # Test standardization
    std_sig = standardize(signal, channel_wise=True)
    assert std_sig.shape == (12, 1000)
    assert np.allclose(std_sig.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(std_sig.std(axis=-1), 1.0, atol=1e-5)
    
    # Test padding
    short_sig = np.random.randn(12, 500)
    padded = pad_or_crop(short_sig, 1000)
    assert padded.shape == (12, 1000)
    assert np.all(padded[:, 500:] == 0.0)
    
    # Test cropping
    long_sig = np.random.randn(12, 1200)
    cropped = pad_or_crop(long_sig, 1000)
    assert cropped.shape == (12, 1000)

    # Test resampling
    resampled = resample_signal(signal, 100.0, 50.0)
    assert resampled.shape == (12, 500)


def test_dataset_caching_and_extraction():
    """Verify that PTBXLDataset initializes, caches data, and retrieves samples."""
    if not os.path.exists(RAW_DIR):
        pytest.skip("PTB-XL raw directory not found.")
        
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    
    # Initialize train dataset (which will trigger caching for all files)
    dataset = PTBXLDataset(
        raw_dir=RAW_DIR,
        cache_dir=CACHE_DIR,
        split="train",
        sampling_rate=100,
        signal_length=1000,
        classes=classes,
        use_caching=True,
        version="1.0.0_test"
    )
    
    # Verify cache files were created
    assert os.path.exists(CACHE_DIR)
    assert os.path.exists(os.path.join(CACHE_DIR, "version.json"))
    assert os.path.exists(os.path.join(CACHE_DIR, "statistics.json"))
    
    # Check dataset length and item structure
    assert len(dataset) > 0
    signal, label = dataset[0]
    
    assert isinstance(signal, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert signal.shape == (12, 1000)
    assert label.shape == (len(classes),)
    
    # Release memory-mapped file handles on Windows before deleting files
    if hasattr(dataset, "cached_signals") and hasattr(dataset.cached_signals, "_mmap"):
        dataset.cached_signals._mmap.close()
    if hasattr(dataset, "cached_labels") and hasattr(dataset.cached_labels, "_mmap"):
        dataset.cached_labels._mmap.close()
    del dataset

    # Clean up test cache
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
