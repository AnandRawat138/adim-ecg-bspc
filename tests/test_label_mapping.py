import os
import pytest
import pandas as pd
from data.loader import validate_dataset_path, load_raw_metadata, map_diagnostic_superclass

RAW_DIR = "C:/Users/anand/.gemini/antigravity/scratch/ecg_classification/data/ptbxl"

def test_validate_dataset_path_failure():
    """Verify that validating an invalid path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        validate_dataset_path("./invalid_path_to_data")


def test_validate_dataset_path_success():
    """Verify that validation passes for the copied PTB-XL folder."""
    if os.path.exists(RAW_DIR):
        validate_dataset_path(RAW_DIR)
    else:
        pytest.skip("PTB-XL raw directory not found at standard path.")


def test_metadata_load_and_mapping():
    """Verify that metadata loads and maps successfully to diagnostic classes."""
    if not os.path.exists(RAW_DIR):
        pytest.skip("PTB-XL raw directory not found.")
        
    df, scp_df = load_raw_metadata(RAW_DIR)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(scp_df, pd.DataFrame)
    assert "scp_codes" in df.columns
    
    target_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    mapped_df = map_diagnostic_superclass(df, scp_df, target_classes)
    
    for cls in target_classes:
        assert cls in mapped_df.columns
        # Check binary labels constraint
        unique_vals = mapped_df[cls].unique()
        assert all(val in [0, 1] for val in unique_vals)
