import numpy as np
import scipy.interpolate

def add_gaussian_noise(signal: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Adds white Gaussian noise to the ECG signal.

    Args:
        signal: ECG signal array of shape (num_leads, signal_length).
        std: Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Augumented ECG signal.
    """
    noise = np.random.normal(0, std, size=signal.shape)
    return signal + noise


def apply_amplitude_scaling(
    signal: np.ndarray, 
    min_scale: float = 0.9, 
    max_scale: float = 1.1
) -> np.ndarray:
    """Randomly scales the amplitude of the signal.

    Args:
        signal: ECG signal array of shape (num_leads, signal_length).
        min_scale: Minimum scale multiplier.
        max_scale: Maximum scale multiplier.

    Returns:
        np.ndarray: Scaled ECG signal.
    """
    scale_factor = np.random.uniform(min_scale, max_scale)
    return signal * scale_factor


def apply_random_time_shift(signal: np.ndarray, max_shift: int = 50) -> np.ndarray:
    """Shifts the signal in time, padding with zeros at the boundaries.

    Args:
        signal: ECG signal of shape (num_leads, signal_length).
        max_shift: Maximum index shift.

    Returns:
        np.ndarray: Shifted ECG signal.
    """
    num_leads, signal_length = signal.shape
    shift = np.random.randint(-max_shift, max_shift + 1)
    
    if shift == 0:
        return signal
        
    shifted = np.zeros_like(signal)
    if shift > 0:
        shifted[:, shift:] = signal[:, :-shift]
    else:
        shifted[:, :shift] = signal[:, -shift:]
    return shifted


def add_baseline_wander(
    signal: np.ndarray, 
    fs: float = 100.0, 
    frequency: float = 0.5, 
    max_amplitude: float = 0.15
) -> np.ndarray:
    """Simulates breathing drift by adding low-frequency sinusoids to each lead.

    Args:
        signal: ECG signal of shape (num_leads, signal_length).
        fs: Sampling rate of the signal in Hz.
        frequency: Baseline wander frequency in Hz (typically < 1.0Hz).
        max_amplitude: Maximum wander amplitude.

    Returns:
        np.ndarray: ECG signal with baseline wander.
    """
    num_leads, signal_length = signal.shape
    t = np.arange(signal_length) / fs
    
    # Generate unique phase/amplitude offsets for each lead to simulate realistic drift
    wander = np.zeros_like(signal)
    for lead in range(num_leads):
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0, max_amplitude)
        wander[lead] = amp * np.sin(2 * np.pi * frequency * t + phase)
        
    return signal + wander


def apply_lead_dropout(signal: np.ndarray, drop_prob: float = 0.15) -> np.ndarray:
    """Sets entire leads to 0 with a given probability. At least one lead is kept.

    Args:
        signal: ECG signal of shape (num_leads, signal_length).
        drop_prob: Probability of dropping a lead.

    Returns:
        np.ndarray: Augmented signal.
    """
    num_leads, signal_length = signal.shape
    mask = np.random.rand(num_leads) > drop_prob
    
    # Ensure we don't drop all leads
    if not np.any(mask):
        mask[np.random.randint(num_leads)] = True
        
    return signal * mask[:, np.newaxis]


def apply_time_masking(signal: np.ndarray, max_mask_length: int = 50) -> np.ndarray:
    """Masks a contiguous time block (sets to zero) across all leads.

    Args:
        signal: ECG signal of shape (num_leads, signal_length).
        max_mask_length: Maximum sample length of the mask.

    Returns:
        np.ndarray: Time-masked signal.
    """
    num_leads, signal_length = signal.shape
    mask_len = np.random.randint(5, max_mask_length)
    if mask_len >= signal_length:
        return signal
        
    start_idx = np.random.randint(0, signal_length - mask_len)
    augmented = signal.copy()
    augmented[:, start_idx : start_idx + mask_len] = 0
    return augmented


def apply_frequency_masking(signal: np.ndarray, max_mask_length: int = 5) -> np.ndarray:
    """Masks frequency bands by computing FFT, zeroing random coefficients, and taking IFFT.

    Args:
        signal: ECG signal of shape (num_leads, signal_length).
        max_mask_length: Maximum frequency bins to zero out.

    Returns:
        np.ndarray: Frequency-masked signal.
    """
    num_leads, signal_length = signal.shape
    
    # Compute FFT along the time dimension
    fft_vals = np.fft.rfft(signal, axis=-1)
    num_freqs = fft_vals.shape[-1]
    
    mask_len = np.random.randint(1, max_mask_length)
    if mask_len >= num_freqs:
        return signal
        
    start_idx = np.random.randint(0, num_freqs - mask_len)
    
    augmented_fft = fft_vals.copy()
    augmented_fft[:, start_idx : start_idx + mask_len] = 0
    
    # Reconstruct signal
    return np.fft.irfft(augmented_fft, n=signal_length, axis=-1)


def apply_lead_permutation(signal: np.ndarray, permute_prob: float = 0.2) -> np.ndarray:
    """Randomly swaps adjacent chest leads within physiological constraints.

    Constraint pairs: (V1, V2), (V3, V4), (V5, V6).
    Standard 12-lead ECG index layout:
    I: 0, II: 1, III: 2, aVR: 3, aVL: 4, aVF: 5,
    V1: 6, V2: 7, V3: 8, V4: 9, V5: 10, V6: 11.

    Args:
        signal: ECG signal of shape (12, signal_length).
        permute_prob: Probability of swap for each pair.

    Returns:
        np.ndarray: Augmented signal.
    """
    if signal.shape[0] != 12:
        return signal # Only applies to standard 12-lead setups
        
    augmented = signal.copy()
    # (V1, V2) swap
    if np.random.rand() < permute_prob:
        augmented[[6, 7]] = augmented[[7, 6]]
    # (V3, V4) swap
    if np.random.rand() < permute_prob:
        augmented[[8, 9]] = augmented[[9, 8]]
    # (V5, V6) swap
    if np.random.rand() < permute_prob:
        augmented[[10, 11]] = augmented[[11, 10]]
        
    return augmented


def apply_time_warping(signal: np.ndarray, warp_factor: float = 0.05) -> np.ndarray:
    """Applies local time stretching/squeezing using cubic spline interpolation.

    Args:
        signal: ECG signal of shape (num_leads, signal_length).
        warp_factor: Severity of time distortion.

    Returns:
        np.ndarray: Time-warped signal.
    """
    num_leads, signal_length = signal.shape
    x = np.arange(signal_length)
    
    # Generate random anchor points for warping path
    num_anchors = 4
    anchor_points = np.linspace(0, signal_length - 1, num_anchors)
    displacements = np.random.normal(0, warp_factor * (signal_length / num_anchors), size=num_anchors)
    displacements[0] = 0
    displacements[-1] = 0
    
    warped_anchors = anchor_points + displacements
    
    # Fit spline to get the new time step indices
    spline = scipy.interpolate.CubicSpline(anchor_points, warped_anchors)
    warped_indices = np.clip(spline(x), 0, signal_length - 1)
    
    warped_signal = np.zeros_like(signal)
    for lead in range(num_leads):
        warped_signal[lead] = np.interp(x, warped_indices, signal[lead])
        
    return warped_signal


def apply_magnitude_warping(signal: np.ndarray, warp_factor: float = 0.05) -> np.ndarray:
    """Applies smooth variations in magnitude over time via a low frequency spline.

    Args:
        signal: ECG signal of shape (num_leads, signal_length).
        warp_factor: Deviation amount from unity scale.

    Returns:
        np.ndarray: Magnitude-warped signal.
    """
    num_leads, signal_length = signal.shape
    x = np.arange(signal_length)
    
    # Spline anchor scaling values
    num_anchors = 4
    anchor_points = np.linspace(0, signal_length - 1, num_anchors)
    scales = np.random.normal(1.0, warp_factor, size=(num_leads, num_anchors))
    
    warped_signal = np.zeros_like(signal)
    for lead in range(num_leads):
        spline = scipy.interpolate.CubicSpline(anchor_points, scales[lead])
        scale_curve = spline(x)
        warped_signal[lead] = signal[lead] * scale_curve
        
    return warped_signal
