import os
import json
import hashlib
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import wfdb
from typing import Tuple, Dict, List, Optional, Any

from data.loader import load_raw_metadata, map_diagnostic_superclass
from data.transforms import standardize, pad_or_crop
import data.augmentations as aug

logger = logging.getLogger(__name__)

class PTBXLDataset(Dataset):
    """PyTorch Dataset for PTB-XL with memory-mapped caching and data augmentations."""

    def __init__(
        self,
        raw_dir: str,
        cache_dir: str,
        split: str = "train",
        sampling_rate: int = 100,
        signal_length: int = 1000,
        classes: Optional[List[str]] = None,
        use_caching: bool = True,
        version: str = "1.0.0",
        augment: bool = False,
        aug_config: Optional[Dict[str, Any]] = None,
        cv_folds: Optional[int] = None,
        fold_idx: Optional[int] = None,
        label_mode: str = "multi",
    ) -> None:
        """Initializes the dataset and loads or constructs the cached arrays.

        Args:
            raw_dir: Directory containing raw PTB-XL files.
            cache_dir: Directory to save memory-mapped files.
            split: Data split ("train", "val", "test", or "all").
            sampling_rate: Sampling frequency, 100 or 500 Hz.
            signal_length: Output signal duration in samples.
            classes: List of diagnostic superclasses.
            use_caching: Enable memmap caching on disk.
            version: Schema version string.
            augment: If True, apply stochastic augmentations.
            aug_config: Dict of augmentation settings.
            cv_folds: Optional total K folds for cross validation overrides.
            fold_idx: Current active validation fold (1-based, e.g. 1 to K) if cv_folds set.
            label_mode: "multi" returns multi-hot vectors over all records; "single" keeps only
                records with exactly one diagnostic superclass and returns the integer class index.
        """
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.split = split.lower()
        self.sampling_rate = sampling_rate
        self.signal_length = signal_length
        # Cast classes to standard python list to prevent json serialization crashes of ListConfig
        self.classes = list(classes) if classes is not None else ["NORM", "MI", "STTC", "CD", "HYP"]
        self.use_caching = use_caching
        self.version = version
        self.augment = augment
        self.aug_config = aug_config or {}
        self.label_mode = label_mode.lower()
        assert self.label_mode in ("multi", "single"), f"Unknown label_mode '{label_mode}'"
        
        # Load raw metadata
        self.df, self.scp_df = load_raw_metadata(self.raw_dir)
        self.df = map_diagnostic_superclass(self.df, self.scp_df, self.classes)
        
        # Contiguous index mapping to handle non-contiguous raw indices
        self.ecg_id_to_idx = {ecg_id: i for i, ecg_id in enumerate(self.df.index)}

        # Set up splits
        self.setup_splits(cv_folds, fold_idx)

        # Caching logic
        if self.use_caching:
            self.setup_cache()
        else:
            logger.warning("Caching is disabled. Signals will be loaded dynamically from raw disk. This is slow.")

    def setup_splits(self, cv_folds: Optional[int], fold_idx: Optional[int]) -> None:
        """Slices the metadata dataframe based on official folds or CV overrides.

        Args:
            cv_folds: Total folds for CV.
            fold_idx: Active fold for validation (1 to cv_folds).
        """
        if cv_folds is not None and fold_idx is not None:
            # Custom K-Fold cross validation split
            assert 1 <= fold_idx <= cv_folds, f"fold_idx must be between 1 and {cv_folds}"
            val_fold = fold_idx
            # Test fold is fixed to 10 as per official, or if cv_folds overrides everything
            # Let's define: Val fold = fold_idx, Test fold = 10, Train = all others < 10
            if cv_folds == 10:
                test_fold = 10
                val_fold = fold_idx
                if val_fold == 10:
                    val_fold = 9  # Shift validation fold if index collides
                train_folds = [f for f in range(1, 11) if f != val_fold and f != test_fold]
            else:
                # Standard CV split
                test_fold = cv_folds
                val_fold = fold_idx
                train_folds = [f for f in range(1, cv_folds + 1) if f != val_fold and f != test_fold]

            if self.split == "train":
                self.split_df = self.df[self.df["strat_fold"].isin(train_folds)]
            elif self.split == "val":
                self.split_df = self.df[self.df["strat_fold"] == val_fold]
            elif self.split == "test":
                self.split_df = self.df[self.df["strat_fold"] == test_fold]
            else:
                self.split_df = self.df
        else:
            # Default official PTB-XL split
            if self.split == "train":
                self.split_df = self.df[self.df["strat_fold"] <= 8]
            elif self.split == "val":
                self.split_df = self.df[self.df["strat_fold"] == 9]
            elif self.split == "test":
                self.split_df = self.df[self.df["strat_fold"] == 10]
            else:
                self.split_df = self.df

        if self.label_mode == "single":
            # Single-label protocol: keep records annotated with exactly one diagnostic superclass
            n_labels = self.split_df[self.classes].sum(axis=1)
            self.split_df = self.split_df[n_labels == 1]

        logger.info(f"Split '{self.split}' created. Total samples: {len(self.split_df)}")
        self.indices = self.split_df.index.values

    def _generate_config_hash(self) -> str:
        """Generates an MD5 signature of current preprocessing parameters."""
        config_dict = {
            "version": self.version,
            "sampling_rate": self.sampling_rate,
            "signal_length": self.signal_length,
            "classes": sorted(self.classes),
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def setup_cache(self) -> None:
        """Preprocesses signals and maps them to disk memory mapped files."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        config_hash = self._generate_config_hash()
        version_file = os.path.join(self.cache_dir, "version.json")
        stats_file = os.path.join(self.cache_dir, "statistics.json")
        
        self.signals_path = os.path.join(self.cache_dir, f"signals_{config_hash}.npy")
        self.labels_path = os.path.join(self.cache_dir, f"labels_{config_hash}.npy")
        
        rebuild_cache = True
        
        if os.path.exists(version_file) and os.path.exists(self.signals_path) and os.path.exists(self.labels_path):
            try:
                with open(version_file, "r") as f:
                    cached_meta = json.load(f)
                if cached_meta.get("hash") == config_hash:
                    rebuild_cache = False
                    logger.info("Found valid preprocessed cache. Loading memmap files.")
            except Exception as e:
                logger.warning(f"Error reading version metadata: {e}. Rebuilding cache.")

        if rebuild_cache:
            logger.info("Cache signature mismatch or missing. Rebuilding preprocessed arrays...")
            self.build_cache(config_hash, version_file, stats_file)
            
        # Map arrays from disk
        self.cached_signals = np.load(self.signals_path, mmap_mode="r")
        self.cached_labels = np.load(self.labels_path, mmap_mode="r")

    def build_cache(self, config_hash: str, version_file: str, stats_file: str) -> None:
        """Loads raw signals, preprocesses, and saves them to a continuous numpy file.

        Args:
            config_hash: Preprocessing parameters configuration hash.
            version_file: Output path for caching version signature.
            stats_file: Output path for saving dataset-wide statistics.
        """
        num_samples = len(self.df)
        num_leads = 12
        
        # Create a valid .npy file with header mapped to disk
        from numpy.lib.format import open_memmap
        temp_signals = open_memmap(
            self.signals_path + ".tmp",
            dtype="float32",
            mode="w+",
            shape=(num_samples, num_leads, self.signal_length)
        )
        
        labels_array = np.zeros((num_samples, len(self.classes)), dtype="float32")
        
        logger.info(f"Preprocessing {num_samples} records. This might take a few minutes...")
        
        # We index signals contiguously using i to prevent out-of-bounds errors on non-contiguous indices
        for i, ecg_id in enumerate(tqdm(self.df.index, desc="Preprocessing signals")):
            row = self.df.loc[ecg_id]
            idx = i
            
            # File path selection
            filename = row["filename_lr"] if self.sampling_rate == 100 else row["filename_hr"]
            filepath = os.path.join(self.raw_dir, filename.replace("/", os.sep))
            
            try:
                # Load record
                record_signal, fields = wfdb.rdsamp(filepath)
                # Transpose to shape (leads, time)
                record_signal = record_signal.T
                
                # Apply transforms
                processed = standardize(record_signal, channel_wise=True)
                processed = pad_or_crop(processed, self.signal_length)
                
                temp_signals[idx] = processed.astype("float32")
            except Exception as e:
                logger.error(f"Failed to load/preprocess record {ecg_id} ({filepath}): {e}")
                # Fallback to zero signal on failure
                temp_signals[idx] = np.zeros((num_leads, self.signal_length), dtype="float32")
            
            # Labels
            labels_array[idx] = row[self.classes].values.astype("float32")
            
        # Flush temp memmap to disk
        temp_signals.flush()
        del temp_signals
        
        # Rename temporary file to final output
        if os.path.exists(self.signals_path):
            os.remove(self.signals_path)
        os.rename(self.signals_path + ".tmp", self.signals_path)
        
        # Save labels
        np.save(self.labels_path, labels_array)
        
        # Calculate statistics
        signals_map = np.load(self.signals_path, mmap_mode="r")
        mean = float(signals_map.mean())
        std = float(signals_map.std())
        class_distribution = {cls: float(self.df[cls].mean()) for cls in self.classes}
        
        stats = {
            "mean": mean,
            "std": std,
            "class_distribution": class_distribution,
            "num_samples": num_samples,
        }
        
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)
            
        # Save version config hash
        version_data = {
            "hash": config_hash,
            "version": self.version,
            "sampling_rate": self.sampling_rate,
            "signal_length": self.signal_length,
            "classes": self.classes,
        }
        with open(version_file, "w") as f:
            json.dump(version_data, f, indent=4)
            
        logger.info(f"Preprocessing completed. Stats saved: {stats}")

    def apply_augmentations(self, signal: np.ndarray) -> np.ndarray:
        """Stochastically applies augmentations defined in config.

        Args:
            signal: ECG signal array of shape (num_leads, signal_length).

        Returns:
            np.ndarray: Augmented signal.
        """
        cfg = self.aug_config
        
        # Time-domain shifts and distortions
        if cfg.get("random_time_shift", {}).get("enabled", False) and np.random.rand() < 0.5:
            max_shift = cfg["random_time_shift"].get("max_shift", 50)
            signal = aug.apply_random_time_shift(signal, max_shift=max_shift)
            
        if cfg.get("time_warping", {}).get("enabled", False) and np.random.rand() < 0.3:
            warp_factor = cfg["time_warping"].get("warp_factor", 0.05)
            signal = aug.apply_time_warping(signal, warp_factor=warp_factor)
            
        # Amplitude / gain scaling
        if cfg.get("amplitude_scaling", {}).get("enabled", False) and np.random.rand() < 0.5:
            min_scale = cfg["amplitude_scaling"].get("min_scale", 0.9)
            max_scale = cfg["amplitude_scaling"].get("max_scale", 1.1)
            signal = aug.apply_amplitude_scaling(signal, min_scale=min_scale, max_scale=max_scale)
            
        if cfg.get("magnitude_warping", {}).get("enabled", False) and np.random.rand() < 0.3:
            warp_factor = cfg["magnitude_warping"].get("warp_factor", 0.05)
            signal = aug.apply_magnitude_warping(signal, warp_factor=warp_factor)

        # Baseline wander / low freq noise
        if cfg.get("baseline_wander", {}).get("enabled", False) and np.random.rand() < 0.4:
            freq = cfg["baseline_wander"].get("frequency", 0.5)
            amp = cfg["baseline_wander"].get("amplitude", 0.05)
            signal = aug.add_baseline_wander(signal, fs=self.sampling_rate, frequency=freq, max_amplitude=amp)

        # High-frequency noise
        if cfg.get("gaussian_noise", {}).get("enabled", False) and np.random.rand() < 0.5:
            std = cfg["gaussian_noise"].get("std", 0.01)
            signal = aug.add_gaussian_noise(signal, std=std)

        # Lead modifications
        if cfg.get("lead_permutation", {}).get("enabled", False) and np.random.rand() < 0.2:
            perm_prob = cfg["lead_permutation"].get("permute_prob", 0.2)
            signal = aug.apply_lead_permutation(signal, permute_prob=perm_prob)

        if cfg.get("lead_dropout", {}).get("enabled", False) and np.random.rand() < 0.2:
            drop_prob = cfg["lead_dropout"].get("drop_prob", 0.15)
            signal = aug.apply_lead_dropout(signal, drop_prob=drop_prob)

        # Masking strategies
        if cfg.get("time_masking", {}).get("enabled", False) and np.random.rand() < 0.3:
            max_len = cfg["time_masking"].get("max_mask_length", 50)
            signal = aug.apply_time_masking(signal, max_mask_length=max_len)

        if cfg.get("frequency_masking", {}).get("enabled", False) and np.random.rand() < 0.3:
            max_len = cfg["frequency_masking"].get("max_mask_length", 5)
            signal = aug.apply_frequency_masking(signal, max_mask_length=max_len)

        return signal

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self):
        # Do not pickle the memory-mapped arrays into DataLoader worker processes; reopen them lazily.
        state = self.__dict__.copy()
        state["cached_signals"] = None
        state["cached_labels"] = None
        return state

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ecg_id = self.indices[idx]
        array_idx = self.ecg_id_to_idx[ecg_id]

        if self.use_caching and getattr(self, "cached_signals", None) is None:
            self.cached_signals = np.load(self.signals_path, mmap_mode="r")
            self.cached_labels = np.load(self.labels_path, mmap_mode="r")

        if self.use_caching:
            signal = self.cached_signals[array_idx].copy()
            label = self.cached_labels[array_idx]
        else:
            # Slow fallback loading
            row = self.split_df.loc[ecg_id]
            filename = row["filename_lr"] if self.sampling_rate == 100 else row["filename_hr"]
            filepath = os.path.join(self.raw_dir, filename.replace("/", os.sep))
            record_signal, _ = wfdb.rdsamp(filepath)
            signal = record_signal.T
            signal = standardize(signal, channel_wise=True)
            signal = pad_or_crop(signal, self.signal_length)
            label = row[self.classes].values.astype("float32")
            
        if self.augment:
            signal = self.apply_augmentations(signal)
            
        if self.label_mode == "single":
            return torch.tensor(signal, dtype=torch.float32), torch.tensor(int(np.argmax(label)), dtype=torch.long)
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
