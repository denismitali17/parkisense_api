import os
import shutil
import librosa
import joblib
import tempfile
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Parkinson's Acoustic Analysis Production API (Optimized)")

# Enable cross-origin requests for frontend connectivity 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model engine variables
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
        
        # Guard check: Ensure the loaded model features match expected inputs
        if hasattr(MODEL, "n_features_in_"):
            EXPECTED_FEATURES = MODEL.n_features_in_
            
        print(f"[SUCCESS] Production ML components loaded. Expected input features: {EXPECTED_FEATURES}")
    except Exception as e:
        print(f"[FATAL] System failed to initialize model assets: {str(e)}")

def extract_single_chunk_features(chunk: np.ndarray, sr: int) -> np.ndarray:
    """Extracts identical clinical handcrafted metrics from an isolated 3-second block"""
    # 1. Fundamental Frequency Tracking
    f0, _, _ = librosa.pyin(chunk, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    
    # 2. Extract Classical Micro-Acoustic Metrics
    jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean) if len(f0_clean) > 1 else 0.0
    rms = librosa.feature.rms(y=chunk)
    shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0.0
    
    harmonic = librosa.effects.harmonic(chunk)
    energy_diff = np.sum((chunk - harmonic)**2)
    hnr = 10 * np.log10(np.sum(harmonic**2) / max(1e-6, energy_diff)) if energy_diff > 0 else 0.0
    
    # 3. Extract Mel-Frequency Cepstral Coefficients
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    
    return np.array([jitter, shimmer, hnr] + list(mfcc_means))

@app.get("/")
def health_check():
    return {"status": "online", "model_integrity": MODEL is not None}

@app.post("/predict")
async def predict_parkinsons(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Prediction engine is currently offline.")
    
    # FIX #3: Strict Audio Format Input Validation Gate
    allowed_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Please upload an audio file ending with {allowed_extensions}."
        )
        
    # FIX #4: Secure OS-Managed Isolated Temporary File Stream
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_path = tmp_file.name
        shutil.copyfileobj(file.file, tmp_file)
        
    try:
        # Load audio with original sample rate preserve rules
        y, sr = librosa.load(temp_path, sr=None)
        
        # Apply Voice Activity Detection (VAD)
        intervals = librosa.effects.split(y, top_db=20)
        v_signal = np.concatenate([y[start:end] for start, end in intervals]) if len(intervals) > 0 else y
        
        # Define samples inside a single standard 3-second block window
        chunk_samples = int(3.0 * sr)
        
        # Pad shorter audio files up to a minimum single chunk length 
        if len(v_signal) < chunk_samples:
            v_signal = np.pad(v_signal, (0, chunk_samples - len(v_signal)), mode='constant')
            
        # FIX #1 & #5: Segment the entire voice stream into sequential 3-second blocks
        chunks_features = []
        step_size = chunk_samples # Non-overlapping steps matching basic block splits
        
        for start_idx in range(0, len(v_signal) - chunk_samples + 1, step_size):
            chunk = v_signal[start_idx : start_idx + chunk_samples]
            feat = extract_single_chunk_features(chunk, sr)
            chunks_features.append(feat)
            
        # Fallback if window parsing boundaries missed segment extractions
        if not chunks_features:
            chunks_features.append(extract_single_chunk_features(v_signal[:chunk_samples], sr))
            
        X_extracted = np.array(chunks_features)
        
        # FIX #2: Structural Dimension Safety Check
        if X_extracted.shape[1] != EXPECTED_FEATURES:
            raise ValueError(f"Feature count mismatch. Model expected {EXPECTED_FEATURES}, but extracted {X_extracted.shape[1]}.")
            
        # Apply training-equivalent clipping and scale matrices across all chunks
        lo, hi = IQR_BOUNDS
        X_clipped = np.clip(X_extracted, lo, hi)
        X_scaled = SCALER.transform(X_clipped)
        
        # Perform predictive inference over all individual windows
        chunk_predictions = MODEL.predict(X_scaled)       # Vector of 0s and 1s per chunk
        chunk_probabilities = MODEL.predict_proba(X_scaled) # Probabilities [[P(0), P(1)], ...] per chunk
        
        # Calculate ensemble averages across the audio timeline
        mean_prob_pd = np.mean(chunk_probabilities[:, 1])  # Average probability for Parkinson's Disease
        
        # Determine overall patient status based on average confidence
        final_prediction = 1 if mean_prob_pd >= 0.5 else 0
        final_confidence = mean_prob_pd if final_prediction == 1 else (1.0 - mean_prob_pd)
        
        return {
            "prediction": final_prediction,
            "diagnosis": "Parkinson's Disease Detected" if final_prediction == 1 else "Healthy Control",
            "confidence_score": round(float(final_confidence), 4),
            "total_chunks_analyzed": len(chunks_features)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")
        
    finally:
        # Securely sweep tracking cleanup path locations
        if os.path.exists(temp_path):
            os.remove(temp_path)