import os
import sys
import io
import traceback
import librosa
import joblib
import numpy as np
import pandas as pd
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

@app.on_event("startup")
def load_assets():
    global MODEL, SCALER, IQR_BOUNDS, EXPECTED_FEATURES
    try:
        MODEL = joblib.load("production_parkinsons_model.joblib")
        SCALER = joblib.load("production_scaler.joblib")
        IQR_BOUNDS = joblib.load("production_iqr_bounds.joblib")
        
        if hasattr(MODEL, "n_features_in_"):
            EXPECTED_FEATURES = MODEL.n_features_in_
        print(f"[SUCCESS] Model assets loaded. Expected features: {EXPECTED_FEATURES}")
    except Exception as e:
        print(f"[FATAL] System failed to initialize model assets: {str(e)}")

def extract_single_chunk_features(chunk: np.ndarray, sr: int) -> np.ndarray:
    """Extracts identical clinical metrics with extreme performance optimizations"""
    
    f0, _, _ = librosa.pyin(
        chunk, 
        fmin=85, 
        fmax=350, 
        sr=sr, 
        hop_length=1024, 
        fill_na=None
    )
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    
    jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean) if len(f0_clean) > 1 else 0.0
    
    rms = librosa.feature.rms(y=chunk, hop_length=1024)
    shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0.0
    
    # Fast HNR calculation bypass to save memory
    harmonic = librosa.effects.harmonic(chunk, margin=2.0)
    energy_diff = np.sum((chunk - harmonic)**2)
    hnr = 10 * np.log10(np.sum(harmonic**2) / max(1e-6, energy_diff)) if energy_diff > 0 else 0.0
    
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13, hop_length=1024)
    mfcc_means = np.mean(mfccs, axis=1)
    
    features = np.array([jitter, shimmer, hnr] + list(mfcc_means))
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

@app.get("/")
def health_check():
    return {"status": "online", "model_integrity": MODEL is not None}

@app.post("/predict")
async def predict_parkinsons(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Prediction engine is offline.")
        
    try:
        # Read raw bytes directly into memory
        file_bytes = await file.read()
        
        # 2. Universal Soundfile memory read 
        try:
            data, sr = sf.read(io.BytesIO(file_bytes))
            y = data.astype(np.float32)
            # Convert stereo to mono instantly if needed
            if len(y.shape) > 1:
                y = y.mean(axis=1)
        except Exception:
            # Read raw binary buffer as float sequence
            y = np.frombuffer(file_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000
            if len(y) == 0:
                raise ValueError("Could not decode audio stream array layout.")

        max_samples = 10 * sr
        if len(y) > max_samples:
            y = y[:max_samples]

        # Standardize Sample Rate
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Silence Truncation
        intervals = librosa.effects.split(y, top_db=20)
        v_signal = np.concatenate([y[start:end] for start, end in intervals]) if len(intervals) > 0 else y
        
        chunk_samples = int(3.0 * sr)
        if len(v_signal) < chunk_samples:
            v_signal = np.pad(v_signal, (0, chunk_samples - len(v_signal)), mode='constant')
            
        chunks_features = []
        
        step_size = chunk_samples
        for start_idx in range(0, len(v_signal) - chunk_samples + 1, step_size):
            chunk = v_signal[start_idx : start_idx + chunk_samples]
            feat = extract_single_chunk_features(chunk, sr)
            chunks_features.append(feat)
            if len(chunks_features) >= 2: 
                break
            
        X_extracted = np.array(chunks_features)
        
        if X_extracted.shape[1] != EXPECTED_FEATURES:
            raise ValueError(f"Feature mismatch. Expected {EXPECTED_FEATURES}, got {X_extracted.shape[1]}.")
            
        lo, hi = IQR_BOUNDS
        X_clipped = np.clip(X_extracted, lo, hi)
        X_scaled = SCALER.transform(X_clipped)
        
        chunk_probabilities = MODEL.predict_proba(X_scaled)
        mean_prob_pd = np.mean(chunk_probabilities[:, 1])
        
        final_prediction = 1 if mean_prob_pd >= 0.5 else 0
        final_confidence = mean_prob_pd if final_prediction == 1 else (1.0 - mean_prob_pd)
        
        return {
            "prediction": final_prediction,
            "diagnosis": "Parkinson's Disease Detected" if final_prediction == 1 else "Healthy Control",
            "confidence_score": round(float(final_confidence), 4),
            "total_chunks_analyzed": len(chunks_features)
        }
        
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("[CRITICAL RUNTIME ERROR TRACEBACK]:")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")