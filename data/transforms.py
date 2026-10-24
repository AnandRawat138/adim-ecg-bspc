import numpy as np
import scipy.signal

def standardize(signal: np.ndarray, channel_wise: bool = True) -> np.ndarray:
    """Standardizes ECG signal by subtracting mean and dividing by standard deviation.

    Args:
        signal: ECG signal array of shape (num_leads, signal_length).
        channel_wise: If True, standardizes each channel independently.

    Returns:
        np.ndarray: Standardized signal of shape (num_leads, signal_length).
    """
    if channel_wise:
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-8
        return (signal - mean) / std
    else:
        mean = signal.mean()
        std = signal.std() + 1e-8
        return (signal - mean) / std


def pad_or_crop(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Pads or crops the signal along the time axis to match target_length.

    Args:
        signal: ECG signal array of shape (num_leads, current_length).
        target_length: Desired length of the time series in samples.

    Returns:
        np.ndarray: Reshaped signal of shape (num_leads, target_length).
    """
    num_leads, current_length = signal.shape
    if current_length == target_length:
        return signal
    
    if current_length < target_length:
        # Zero padding
        pad_width = target_length - current_length
        padded = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
        return padded
    else:
        # Center crop
        start_idx = (current_length - target_length) // 2
        cropped = signal[:, start_idx : start_idx + target_length]
        return cropped


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resamples the ECG signal from original frequency to target frequency.

    Args:
        signal: ECG signal array of shape (num_leads, signal_length).
        orig_fs: Original sampling frequency in Hz.
        target_fs: Target sampling frequency in Hz.

    Returns:
        np.ndarray: Resampled signal.
    """
    if orig_fs == target_fs:
        return signal
    
    num_leads, current_length = signal.shape
    new_length = int(current_length * target_fs / orig_fs)
    resampled = scipy.signal.resample(signal, new_length, axis=-1)
    return resampled
