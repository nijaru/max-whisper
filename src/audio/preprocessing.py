"""
Basic audio preprocessing pipeline.
This will serve as our Python baseline before porting to Mojo.
"""

import numpy as np
from typing import Tuple, Optional


def load_audio(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    For now, this is a placeholder - we'll implement actual audio loading later.
    """
    # Placeholder: generate synthetic audio for testing
    duration = 5.0  # 5 seconds
    samples = int(duration * sample_rate)
    
    # Generate a simple sine wave for testing
    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    return audio


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80
) -> np.ndarray:
    """
    Compute mel-spectrogram from audio.
    This is our CPU baseline implementation.
    """
    # Window function
    window = np.hanning(n_fft)
    
    # Pad audio
    audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')
    
    # Compute STFT
    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = audio_padded[start:start + n_fft] * window
        fft_frame = np.fft.rfft(frame)
        stft[:, i] = fft_frame
    
    # Magnitude spectrogram
    magnitude = np.abs(stft) ** 2
    
    # Mel filter bank (simplified)
    mel_filters = create_mel_filters(sample_rate, n_fft, n_mels)
    mel_spec = mel_filters @ magnitude
    
    # Log scale
    log_mel_spec = np.log10(np.maximum(mel_spec, 1e-10))
    
    return log_mel_spec


def create_mel_filters(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """
    Create mel filter bank.
    Simplified implementation for baseline.
    """
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)
    
    # Frequency range
    low_freq_mel = 0
    high_freq_mel = hz_to_mel(sample_rate // 2)
    
    # Mel points
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # FFT bin points
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # Create filter bank
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    
    return fbank


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize mel-spectrogram features to match Whisper's expected input.
    """
    # Center and scale to roughly [-1, 1] range
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    normalized = (features - mean) / (std + 1e-8)
    
    return normalized


def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Complete audio preprocessing pipeline.
    This is our baseline implementation that we'll optimize with Mojo.
    """
    # Load audio
    audio = load_audio(file_path)
    
    # Compute mel-spectrogram
    mel_spec = compute_mel_spectrogram(audio)
    
    # Normalize features
    features = normalize_features(mel_spec)
    
    return features


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing audio preprocessing pipeline...")
    
    # Test with synthetic audio
    features = preprocess_audio("dummy_path.wav")
    print(f"Mel-spectrogram shape: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print("Audio preprocessing baseline ready!")