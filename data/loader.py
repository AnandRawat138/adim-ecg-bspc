import os
import ast
import logging
import pandas as pd
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

def validate_dataset_path(raw_dir: str) -> None:
    """Verifies that PTB-XL files and directories exist.

    Args:
        raw_dir: Path to the raw PTB-XL database directory.

    Raises:
        FileNotFoundError: If database CSV files or waveform folders are missing.
    """
    db_csv = os.path.join(raw_dir, "ptbxl_database.csv")
    scp_csv = os.path.join(raw_dir, "scp_statements.csv")
    records_100 = os.path.join(raw_dir, "records100")
    records_500 = os.path.join(raw_dir, "records500")

    if not os.path.exists(db_csv):
        raise FileNotFoundError(
            f"Missing metadata file: '{db_csv}'. "
            f"Please verify that the PTB-XL database is placed correctly."
        )

    if not os.path.exists(scp_csv):
        raise FileNotFoundError(
            f"Missing statements mapping file: '{scp_csv}'. "
            f"Please verify that the PTB-XL database is placed correctly."
        )

    if not os.path.exists(records_100) and not os.path.exists(records_500):
        raise FileNotFoundError(
            f"Waveform directories ('records100' or 'records500') not found in '{raw_dir}'. "
            f"Ensure raw signal folders have been unpacked."
        )

    logger.info("PTB-XL raw dataset path successfully validated.")


def load_raw_metadata(raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads PTB-XL database metadata and SCP statement CSV mappings.

    Args:
        raw_dir: Raw dataset directory path.

    Returns:
        Tuple containing:
            - df (pd.DataFrame): PTB-XL metadata table.
            - scp_df (pd.DataFrame): SCP statement mappings table.
    """
    validate_dataset_path(raw_dir)
    
    db_csv = os.path.join(raw_dir, "ptbxl_database.csv")
    scp_csv = os.path.join(raw_dir, "scp_statements.csv")
    
    logger.info(f"Loading metadata from {db_csv}")
    df = pd.read_csv(db_csv, index_col="ecg_id")
    # Convert string representation of dict to dict objects
    df["scp_codes"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x))

    logger.info(f"Loading SCP mapping statements from {scp_csv}")
    scp_df = pd.read_csv(scp_csv, index_col=0)
    
    return df, scp_df


def map_diagnostic_superclass(
    df: pd.DataFrame, 
    scp_df: pd.DataFrame,
    target_classes: List[str]
) -> pd.DataFrame:
    """Groups SCP labels into 5 clinical diagnostic superclasses.

    The 5 superclasses are: NORM, MI, STTC, CD, HYP.
    
    Args:
        df: Metadata dataframe.
        scp_df: SCP statement mapping dataframe.
        target_classes: List of superclasses to retain (default: ['NORM', 'MI', 'STTC', 'CD', 'HYP'])

    Returns:
        pd.DataFrame: Modified dataframe with diagnostic binary indicators.
    """
    # Map statements to diagnostic classes
    diag_statements = scp_df[scp_df["diagnostic"] == 1]
    
    def get_superclasses(scp_dict: Dict[str, float]) -> List[str]:
        superclasses = set()
        for key in scp_dict.keys():
            if key in diag_statements.index:
                superclass = diag_statements.loc[key, "diagnostic_class"]
                if pd.notna(superclass) and superclass in target_classes:
                    superclasses.add(superclass)
        return list(superclasses)

    df["diagnostic_superclass"] = df["scp_codes"].apply(get_superclasses)
    
    # Add binary multi-label indicator columns for target classes
    for cls in target_classes:
        df[cls] = df["diagnostic_superclass"].apply(lambda x: 1 if cls in x else 0)

    # Calculate basic stats
    class_sums = df[target_classes].sum()
    logger.info(f"Class distribution after mapping: \n{class_sums.to_string()}")
    
    return df
