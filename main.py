import os
import sys
import io
import traceback
from pathlib import Path
import librosa
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import parselmouth
from parselmouth.praat import call
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["LIBROSA_CACHE_DIR"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = FastAPI(title="Parkinson's Acoustic Analysis Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
SCALER = None
IQR_BOUNDS = None
DEEP_MODEL = None
NORM_BOUNDS = None
MODEL_TYPE = "classical"  # "classical" or "deep"
EXPECTED_FEATURES = 16
POSITIVE_CLASS_LABEL = int(os.getenv("PARKINSONS_POSITIVE_CLASS", "1"))
PREDICTION_THRESHOLD = float(os.getenv("PARKINSONS_THRESHOLD", "0.5"))

BASE_DIR = Path(__file__).resolve().parent

# Updated production path targets matching your 4 winning deployment assets
MODEL_PATH = BASE_DIR / "best_classical_svm_model.joblib"
SCALER_PATH = BASE_DIR / "classical_features_scaler.joblib"
IQR_BOUNDS_PATH = BASE_DIR / "production_iqr_bounds.joblib"  # Optional fallback if kept
DEEP_MODEL_PATH = BASE_DIR / "best_deep_learning_crnn_model.keras"
NORM_BOUNDS_PATH = BASE_DIR / "deep_learning_normalization_bounds.joblib"

@app.on_event("startup")
def load_assets():  
    """Initializes and loads pre-trained machine learning artifacts upon server startup"""
    global MODEL, SCALER, IQR_BOUNDS, EXPECTED_FEATURES, DEEP_MODEL, MODEL_TYPE, NORM_BOUNDS
    try:
        # Load deep learning model and scaling parameters
        if DEEP_MODEL_PATH.exists():
            DEEP_MODEL = tf.keras.models.load_model(DEEP_MODEL_PATH, compile=False)
            MODEL_TYPE = "deep"
            print(f"[SUCCESS] Leaderboard-winning CRNN model loaded from {DEEP_MODEL_PATH}")
            
            if NORM_BOUNDS_PATH.exists():
                NORM_BOUNDS = joblib.load(NORM_BOUNDS_PATH)
                print(f"[SUCCESS] Deep learning normalization bounds loaded.")
        else:
            print(f"[WARN] Deep learning model file not found at {DEEP_MODEL_PATH}. Checking classical fallbacks...")

        # Load classical model pipeline components
        if MODEL_PATH.exists():
            MODEL = joblib.load(MODEL_PATH)
            if SCALER_PATH.exists():
                SCALER = joblib.load(SCALER_PATH)
            if IQR_BOUNDS_PATH.exists():
                IQR_BOUNDS = joblib.load(IQR_BOUNDS_PATH)

            if hasattr(MODEL, "n_features_in_"):
                EXPECTED_FEATURES = MODEL.n_features_in_
            
            if DEEP_MODEL is None:
                MODEL_TYPE = "classical"
            print(f"[SUCCESS] Classical SVM model assets loaded successfully. Expected features: {EXPECTED_FEATURES}")
            
    except Exception as e:
        print(f"[FATAL] System failed to initialize machine learning assets: {str(e)}")

def extract_log_mel_tensor(chunk: np.ndarray, sr: int, n_mels: int = 128, target_shape: tuple = (128, 128)) -> np.ndarray:
    """Maps continuous time-domain signals to 2D structural logarithmic frequency scale representations"""
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

def extract_single_chunk_features(chunk: np.ndarray, sr: int) -> np.ndarray:
    """Extracts exact clinical metrics using parselmouth (Praat) to match training pipeline"""
    try:
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
    except Exception:
        local_jitter = 0.0
        local_shimmer = 0.0
        mean_hnr = 0.0

    peak = np.max(np.abs(chunk)) if len(chunk) > 0 else 0.0
    if peak > 0.0:
        chunk_normalized = chunk / peak
    else:
        chunk_normalized = chunk

    mfccs = librosa.feature.mfcc(y=chunk_normalized, sr=sr, n_mfcc=13, hop_length=512)
    mfcc_means = np.mean(mfccs, axis=1)

    features = np.array([local_jitter, local_shimmer, mean_hnr] + list(mfcc_means))
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

def _resolve_positive_probability(model, probabilities):
    """Map model output probabilities to the positive class without assuming a fixed column."""
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) == 0:
        return probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

    positive_class = int(os.getenv("PARKINSONS_POSITIVE_CLASS", str(classes[-1])))
    if positive_class in classes:
        positive_index = int(np.where(classes == positive_class)[0][0])
        return probabilities[:, positive_index]

    return probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

def predict_from_audio_bytes(file_bytes: bytes):
    if MODEL is None and DEEP_MODEL is None:
        raise RuntimeError("Prediction engine is offline.")

    y = None
    sr = 16000

    try:
        data, sr = sf.read(io.BytesIO(file_bytes))
        y = data.astype(np.float32)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
    except Exception:
        y = np.frombuffer(file_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sr = 16000
        if len(y) == 0:
            raise ValueError("Could not decode audio stream byte vector array structure.")

    max_samples = 10 * sr
    if len(y) > max_samples:
        y = y[:max_samples]

    if sr != 16000:
        try:
            if len(y) > 512:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            else:
                xp = np.arange(len(y))
                x_new = np.linspace(0, len(y) - 1, int(len(y) * 16000 / sr))
                y = np.interp(x_new, xp, y).astype(np.float32)
        except Exception:
            pass
        sr = 16000

    chunk_samples = int(3.0 * sr)

    try:
        intervals = librosa.effects.split(y, top_db=20)
        if len(intervals) > 0:
            v_signal = np.concatenate([y[start:end] for start, end in intervals])
            if len(v_signal) < chunk_samples:
                v_signal = y
        else:
            v_signal = y
    except Exception:
        v_signal = y

    peak = float(np.max(np.abs(v_signal))) if len(v_signal) > 0 else 0.0
    if peak > 0.0:
        v_signal = v_signal / peak

    if len(v_signal) < chunk_samples:
        v_signal = np.pad(v_signal, (0, chunk_samples - len(v_signal)), mode='constant')

    if MODEL_TYPE == "deep":
        return _predict_deep_learning(v_signal, sr, chunk_samples)
    else:
        return _predict_classical(v_signal, sr, chunk_samples)

def _predict_classical(v_signal: np.ndarray, sr: int, chunk_samples: int):
    """Classical ML prediction path using handcrafted features"""
    chunks_features = []
    step_size = max(1, chunk_samples // 2)

    for start_idx in range(0, len(v_signal) - chunk_samples + 1, step_size):
        chunk = v_signal[start_idx:start_idx + chunk_samples]
        if len(chunk) < chunk_samples:
            continue

        try:
            feat = extract_single_chunk_features(chunk, sr)
            chunks_features.append(feat)
        except Exception:
            continue

    if not chunks_features:
        fallback_chunk = v_signal[:chunk_samples]
        if len(fallback_chunk) < chunk_samples:
            fallback_chunk = np.pad(fallback_chunk, (0, chunk_samples - len(fallback_chunk)), mode='constant')
        chunks_features.append(extract_single_chunk_features(fallback_chunk, sr))

    X_extracted = np.array(chunks_features)
    if X_extracted.ndim == 1:
        X_extracted = X_extracted.reshape(1, -1)
    elif X_extracted.shape[0] > 1:
        X_extracted = np.mean(X_extracted, axis=0, keepdims=True)

    if X_extracted.shape[1] != EXPECTED_FEATURES:
        raise ValueError(f"Feature shape tracking mismatch. Expected {EXPECTED_FEATURES}, processed {X_extracted.shape[1]}.")

    # Gracefully bypass clipping if IQR bounds are omitted for the final SVM pipeline
    if IQR_BOUNDS is not None:
        lo, hi = IQR_BOUNDS
        X_extracted = np.clip(X_extracted, lo, hi)
        
    X_scaled = SCALER.transform(X_extracted)

    chunk_probabilities = MODEL.predict_proba(X_scaled)
    positive_probabilities = _resolve_positive_probability(MODEL, chunk_probabilities)
    mean_positive_prob = float(np.mean(positive_probabilities))
    max_positive_prob = float(np.max(positive_probabilities))
    decision_probability = max_positive_prob if max_positive_prob >= mean_positive_prob else mean_positive_prob

    final_prediction = 1 if decision_probability >= PREDICTION_THRESHOLD else 0
    final_confidence = decision_probability if final_prediction == 1 else (1.0 - decision_probability)

    return {
        "prediction": final_prediction,
        "diagnosis": "Parkinson's Disease Detected" if final_prediction == 1 else "Healthy Control",
        "confidence_score": round(float(final_confidence), 4),
        "total_chunks_analyzed": len(chunks_features),
        "positive_class_probability": round(decision_probability, 4),
        "positive_class_label": int(POSITIVE_CLASS_LABEL),
        "model_type": "classical"
    }

def _predict_deep_learning(v_signal: np.ndarray, sr: int, chunk_samples: int):
    """Deep learning prediction path using spectrogram features"""
    chunks_spectrograms = []
    step_size = max(1, chunk_samples // 2)

    for start_idx in range(0, len(v_signal) - chunk_samples + 1, step_size):
        chunk = v_signal[start_idx:start_idx + chunk_samples]
        if len(chunk) < chunk_samples:
            continue

        try:
            spec = extract_log_mel_tensor(chunk, sr)
            chunks_spectrograms.append(spec)
        except Exception:
            continue

    if not chunks_spectrograms:
        fallback_chunk = v_signal[:chunk_samples]
        if len(fallback_chunk) < chunk_samples:
            fallback_chunk = np.pad(fallback_chunk, (0, chunk_samples - len(fallback_chunk)), mode='constant')
        chunks_spectrograms.append(extract_log_mel_tensor(fallback_chunk, sr))

    X_extracted = np.array(chunks_spectrograms)
    
    if NORM_BOUNDS is not None:
        t_min, t_max = NORM_BOUNDS["t_min"], NORM_BOUNDS["t_max"]
    else:
        t_min, t_max = X_extracted.min(), X_extracted.max()
        
    denom = (t_max - t_min + 1e-7)
    X_normalized = np.clip((X_extracted - t_min) / denom, 0.0, 1.0)
    
    expected_channels = DEEP_MODEL.input_shape[-1] if hasattr(DEEP_MODEL, "input_shape") else 1
    if expected_channels == 3 and X_normalized.shape[-1] == 1:
        X_input = np.repeat(X_normalized, 3, axis=-1)
    elif expected_channels == 1 and X_normalized.shape[-1] == 3:
        X_input = X_normalized[..., :1]
    else:
        X_input = X_normalized

    # Get batch predictions safely
    chunk_probabilities = DEEP_MODEL.predict(X_input, verbose=0).ravel()
    mean_positive_prob = float(np.mean(chunk_probabilities))
    max_positive_prob = float(np.max(chunk_probabilities))
    decision_probability = max_positive_prob if max_positive_prob >= mean_positive_prob else mean_positive_prob

    final_prediction = 1 if decision_probability >= PREDICTION_THRESHOLD else 0
    final_confidence = decision_probability if final_prediction == 1 else (1.0 - decision_probability)

    return {
        "prediction": final_prediction,
        "diagnosis": "Parkinson's Disease Detected" if final_prediction == 1 else "Healthy Control",
        "confidence_score": round(float(final_confidence), 4),
        "total_chunks_analyzed": len(chunks_spectrograms),
        "positive_class_probability": round(decision_probability, 4),
        "positive_class_label": int(POSITIVE_CLASS_LABEL),
        "model_type": "deep_learning"
    }

@app.get("/")
def health_check():
    """Simple status route verification utility"""
    return {"status": "online", "model_integrity": MODEL is not None or DEEP_MODEL is not None, "model_type": MODEL_TYPE}

@app.post("/predict")
async def predict_parkinsons(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        return predict_from_audio_bytes(file_bytes)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("[CRITICAL RUNTIME ERROR TRACEBACK]:")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")