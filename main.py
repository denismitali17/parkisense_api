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
EXPECTED_FEATURES = 16
POSITIVE_CLASS_LABEL = int(os.getenv("PARKINSONS_POSITIVE_CLASS", "1"))
PREDICTION_THRESHOLD = float(os.getenv("PARKINSONS_THRESHOLD", "0.5"))
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "production_parkinsons_model.joblib"
SCALER_PATH = BASE_DIR / "production_scaler.joblib"
IQR_BOUNDS_PATH = BASE_DIR / "production_iqr_bounds.joblib"

@app.on_event("startup")
def load_assets():  
    """Initializes and loads pre-trained machine learning artifacts upon server startup"""
    global MODEL, SCALER, IQR_BOUNDS, EXPECTED_FEATURES
    try:
        MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        IQR_BOUNDS = joblib.load(IQR_BOUNDS_PATH)

        if hasattr(MODEL, "n_features_in_"):
            EXPECTED_FEATURES = MODEL.n_features_in_
        print(f"[SUCCESS] Model assets loaded successfully. Expected features: {EXPECTED_FEATURES}")
    except Exception as e:
        print(f"[FATAL] System failed to initialize machine learning assets: {str(e)}")

def extract_single_chunk_features(chunk: np.ndarray, sr: int) -> np.ndarray:
    """Extracts identical clinical metrics with optimized frame-stepping configurations"""
    

    try:
        f0, _, _ = librosa.pyin(
            chunk, 
            fmin=85, 
            fmax=350, 
            sr=sr, 
            hop_length=512, 
            fill_na=None
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    except Exception:

        f0_clean = np.array([])
    
    # Classical Micro-Acoustic Metrics (Jitter & Shimmer)
    jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean) if (len(f0_clean) > 1 and np.mean(f0_clean) > 0) else 0.0
    
    rms = librosa.feature.rms(y=chunk, hop_length=1024)
    shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0.0
    
    # Harmonic-to-Noise Ratio (HNR) Logic
    try:
        harmonic = librosa.effects.harmonic(chunk, margin=2.0)
        energy_diff = np.sum((chunk - harmonic)**2)
        hnr = 10 * np.log10(np.sum(harmonic**2) / max(1e-6, energy_diff)) if energy_diff > 0 else 0.0
    except Exception:
        hnr = 0.0
    
    # Extract 13 Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13, hop_length=1024)
    mfcc_means = np.mean(mfccs, axis=1)
    
    # Concatenate and normalize structural arrays against math edge cases
    features = np.array([jitter, shimmer, hnr] + list(mfcc_means))
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
    if MODEL is None:
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
        intervals = librosa.effects.split(y, top_db=25)
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

    chunks_features = []
    step_size = chunk_samples

    for start_idx in range(0, len(v_signal) - chunk_samples + 1, step_size):
        chunk = v_signal[start_idx:start_idx + chunk_samples]
        if len(chunk) < chunk_samples:
            continue

        try:
            feat = extract_single_chunk_features(chunk, sr)
            chunks_features.append(feat)
        except Exception:
            continue

        if len(chunks_features) >= 2:
            break

    if not chunks_features:
        fallback_chunk = v_signal[:chunk_samples]
        if len(fallback_chunk) < chunk_samples:
            fallback_chunk = np.pad(fallback_chunk, (0, chunk_samples - len(fallback_chunk)), mode='constant')
        chunks_features.append(extract_single_chunk_features(fallback_chunk, sr))

    X_extracted = np.array(chunks_features)
    if X_extracted.shape[1] != EXPECTED_FEATURES:
        raise ValueError(f"Feature shape tracking mismatch. Expected {EXPECTED_FEATURES}, processed {X_extracted.shape[1]}.")

    lo, hi = IQR_BOUNDS
    X_clipped = np.clip(X_extracted, lo, hi)
    X_scaled = SCALER.transform(X_clipped)

    chunk_probabilities = MODEL.predict_proba(X_scaled)
    positive_probabilities = _resolve_positive_probability(MODEL, chunk_probabilities)
    mean_positive_prob = float(np.mean(positive_probabilities))

    final_prediction = 1 if mean_positive_prob >= PREDICTION_THRESHOLD else 0
    final_confidence = mean_positive_prob if final_prediction == 1 else (1.0 - mean_positive_prob)

    return {
        "prediction": final_prediction,
        "diagnosis": "Parkinson's Disease Detected" if final_prediction == 1 else "Healthy Control",
        "confidence_score": round(float(final_confidence), 4),
        "total_chunks_analyzed": len(chunks_features),
        "positive_class_probability": round(mean_positive_prob, 4),
        "positive_class_label": int(POSITIVE_CLASS_LABEL),
    }


@app.get("/")
def health_check():
    """Simple status route verification utility"""
    return {"status": "online", "model_integrity": MODEL is not None}


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