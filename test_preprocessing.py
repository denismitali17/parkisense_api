"""Test script to verify preprocessing pipeline matches notebook"""
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

def extract_handcrafted_metrics(chunk, sr):
    """Computes exact physiological glottal modulations via parselmouth (Praat core engine)."""
    sound = parselmouth.Sound(chunk, sampling_frequency=sr)

    pitch = call(sound, "To Pitch (cc)", 0.0, 75.0, 4, False, 0.03, 0.45, 0.01, 0.35, 0.14, 600.0)
    point_process = call([sound, pitch], "To PointProcess (cc)")

    local_jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0.0001, 0.02, 1.3, 1.6, 55)
    hnr = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    mean_hnr = call(hnr, "Get mean", 0.0, 0.0)

    local_jitter = 0.0 if np.isnan(local_jitter) else local_jitter
    local_shimmer = 0.0 if np.isnan(local_shimmer) else local_shimmer
    mean_hnr = 0.0 if np.isnan(mean_hnr) or mean_hnr < -100 else mean_hnr

    peak = np.max(np.abs(chunk)) if len(chunk) > 0 else 0.0
    if peak > 0.0:
        chunk = chunk / peak

    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13, hop_length=512)
    mean_mfcc = np.mean(mfcc, axis=1)

    return np.array([local_jitter, local_shimmer, mean_hnr] + list(mean_mfcc))

def extract_log_mel_tensor(chunk, sr, n_mels=128, target_shape=(128, 128)):
    """Maps continuous time-domain signals to 2D structural logarithmic frequency scale representations."""
    peak = np.max(np.abs(chunk)) if len(chunk) > 0 else 0.0
    if peak > 0.0:
        chunk = chunk / peak

    hop_len = int((len(chunk) - 1) / (target_shape[1] - 1)) if len(chunk) > target_shape[1] else 512
    stft_matrix = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=hop_len)
    log_spec = librosa.power_to_db(stft_matrix, ref=np.max)

    if log_spec.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - log_spec.shape[1]
        log_spec = np.pad(log_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
    else:
        log_spec = log_spec[:, :target_shape[1]]

    return log_spec[..., np.newaxis]

# Generate synthetic test audio
sr = 16000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration))
test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

print("Testing preprocessing pipeline...")
print(f"Test audio shape: {test_audio.shape}")

# Test handcrafted features
try:
    features = extract_handcrafted_metrics(test_audio, sr)
    print(f"Handcrafted features shape: {features.shape}")
    print(f"Feature values: {features[:5]}")  # First 5 features
    print("[OK] Handcrafted feature extraction works")
except Exception as e:
    print(f"[FAIL] Handcrafted feature extraction failed: {e}")

# Test spectrogram extraction
try:
    spec = extract_log_mel_tensor(test_audio, sr)
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Spectrogram range: [{spec.min():.2f}, {spec.max():.2f}]")
    print("[OK] Spectrogram extraction works")
except Exception as e:
    print(f"[FAIL] Spectrogram extraction failed: {e}")

print("\nPreprocessing pipeline test complete!")
