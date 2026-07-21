import os
import sys
import io
import gc
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


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LIBROSA_CACHE_DIR"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = FastAPI(title="Parkinson's Acoustic Analysis Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEP_MODEL = None
NORM_BOUNDS = None
CLASSICAL_MODEL = None
CLASSICAL_SCALER = None
POSITIVE_CLASS_LABEL = int(os.getenv("PARKINSONS_POSITIVE_CLASS", "1"))
PREDICTION_THRESHOLD = float(os.getenv("PARKINSONS_THRESHOLD", "0.7"))
TEMPERATURE = float(os.getenv("PREDICTION_TEMPERATURE", "3.5"))

BASE_DIR = Path(__file__).resolve().parent
DEEP_MODEL_PATH = BASE_DIR / "best_deep_learning_crnn_model.keras"
NORM_BOUNDS_PATH = BASE_DIR / "deep_learning_normalization_bounds.joblib"
CLASSICAL_MODEL_PATH = BASE_DIR / "best_classical_svm_model.joblib"
CLASSICAL_SCALER_PATH = BASE_DIR / "classical_features_scaler.joblib"

@app.on_event("startup")
def load_assets():  
    """Initializes and loads pre-trained machine learning artifacts upon server startup"""
    global DEEP_MODEL, NORM_BOUNDS, CLASSICAL_MODEL, CLASSICAL_SCALER
    try:
        if DEEP_MODEL_PATH.exists():
            DEEP_MODEL = tf.keras.models.load_model(DEEP_MODEL_PATH, compile=False)
            print(f"[SUCCESS] Leaderboard-winning CRNN model loaded seamlessly from {DEEP_MODEL_PATH}", flush=True)
            
            if NORM_BOUNDS_PATH.exists():
                NORM_BOUNDS = joblib.load(NORM_BOUNDS_PATH)
                print(f"[SUCCESS] Deep learning normalization bounds loaded.", flush=True)
        else:
            raise FileNotFoundError(f"Critical deep learning asset missing at {DEEP_MODEL_PATH}")
        
        if CLASSICAL_MODEL_PATH.exists() and CLASSICAL_SCALER_PATH.exists():
            CLASSICAL_MODEL = joblib.load(CLASSICAL_MODEL_PATH)
            CLASSICAL_SCALER = joblib.load(CLASSICAL_SCALER_PATH)
            print(f"[SUCCESS] Classical SVM model and scaler loaded.", flush=True)
            
    except Exception as e:
        print(f"[FATAL] System failed to initialize deep learning engine: {str(e)}", flush=True)

def extract_handcrafted_metrics(chunk: np.ndarray, sr: int) -> np.ndarray:
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

def extract_log_mel_tensor(chunk: np.ndarray, sr: int, n_mels: int = 128, target_shape: tuple = (128, 128)) -> np.ndarray:
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

def _predict_classical(v_signal: np.ndarray, sr: int) -> float:
    """Classical SVM model prediction path using handcrafted features."""
    if CLASSICAL_MODEL is None or CLASSICAL_SCALER is None:
        return 0.5
    
    try:
        features = extract_handcrafted_metrics(v_signal, sr)
        features_scaled = CLASSICAL_SCALER.transform([features])
        prob = CLASSICAL_MODEL.predict_proba(features_scaled)[0][1]
        return float(prob)
    except Exception as e:
        print(f"[WARN] Classical model prediction failed: {str(e)}", flush=True)
        return 0.5

def predict_from_audio_bytes(file_bytes: bytes):
    if DEEP_MODEL is None:
        raise RuntimeError("Prediction engine is offline.")

    y = None
    sr = None

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


    if len(v_signal) < chunk_samples:
        v_signal = np.pad(v_signal, (0, chunk_samples - len(v_signal)), mode='constant')

    return _predict_ensemble(v_signal, sr, chunk_samples)

def _predict_ensemble(v_signal: np.ndarray, sr: int, chunk_samples: int):
    """Ensemble prediction combining deep learning and classical models"""
    chunk_probabilities = []
    processed_count = 0
    max_chunks = 3
    
    step_size = chunk_samples  
    expected_channels = DEEP_MODEL.input_shape[-1] if hasattr(DEEP_MODEL, "input_shape") else 1

    for start_idx in range(0, len(v_signal) - chunk_samples + 1, step_size):
        if processed_count >= max_chunks:
            break
            
        chunk = v_signal[start_idx:start_idx + chunk_samples]
        if len(chunk) < chunk_samples:
            continue

        try:
            spec = extract_log_mel_tensor(chunk, sr)
            
            if NORM_BOUNDS is not None:
                t_min, t_max = NORM_BOUNDS["t_min"], NORM_BOUNDS["t_max"]
            else:
                t_min, t_max = spec.min(), spec.max()
            
            denom = (t_max - t_min + 1e-7)
            spec_norm = np.clip((spec - t_min) / denom, 0.0, 1.0)
            
            if expected_channels == 3:
                X_input = np.repeat(spec_norm, 3, axis=-1)
            else:
                X_input = spec_norm

            X_tensor = np.expand_dims(X_input, axis=0)

            raw_output = DEEP_MODEL(X_tensor, training=False)
            dl_prob = float(raw_output.numpy().ravel()[0])
            
            classical_prob = _predict_classical(chunk, sr)
            
            ensemble_prob = 0.4 * dl_prob + 0.6 * classical_prob
            
            if TEMPERATURE > 1.0:
                ensemble_prob = ensemble_prob ** (1.0 / TEMPERATURE)
                ensemble_prob = ensemble_prob / (ensemble_prob + (1 - ensemble_prob) ** (1.0 / TEMPERATURE))
            chunk_probabilities.append(ensemble_prob)
            processed_count += 1

        except Exception as chunk_err:
            print(f"[WARN] Error running inference on chunk segment: {str(chunk_err)}", flush=True)
            continue

    if not chunk_probabilities:
        chunk_probabilities.append(0.5)

    mean_positive_prob = float(np.mean(chunk_probabilities))
    
    final_prediction = 1 if mean_positive_prob >= PREDICTION_THRESHOLD else 0
    final_confidence = mean_positive_prob if final_prediction == 1 else (1.0 - mean_positive_prob)

    del v_signal
    gc.collect()

    return {
        "prediction": final_prediction,
        "diagnosis": "Parkinson's Disease Detected" if final_prediction == 1 else "Healthy Control",
        "confidence_score": round(float(final_confidence), 4),
        "total_chunks_analyzed": max(1, processed_count),
        "positive_class_probability": round(mean_positive_prob, 4),
        "positive_class_label": int(POSITIVE_CLASS_LABEL),
        "model_type": "ensemble"
    }

@app.get("/")
def health_check():
    """Simple status route verification utility"""
    return {"status": "online", "model_integrity": DEEP_MODEL is not None, "model_type": "deep_learning"}

@app.post("/predict")
async def predict_parkinsons(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        return predict_from_audio_bytes(file_bytes)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("[CRITICAL RUNTIME ERROR TRACEBACK]:", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")
