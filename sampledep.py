import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
import wfdb.processing as wp
from scipy.signal import butter, filtfilt, resample
from collections import Counter

# ===============================
# Utility functions
# ===============================
def bandpass_filter(signal, fs=360, low=0.5, high=40):
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def normalize_beats(beats):
    beats = beats - np.mean(beats, axis=1, keepdims=True)
    beats = beats / (np.std(beats, axis=1, keepdims=True) + 1e-8)
    return beats

def extract_beats(signal, r_peaks, window=180):
    beats = []
    for r in r_peaks:
        if r - window >= 0 and r + window < len(signal):
            beats.append(signal[r-window:r+window])
    return np.array(beats)

def resize_beats(beats, target_length=250):
    """Resample each beat to target length"""
    from scipy.signal import resample
    if beats.shape[1] == target_length:
        return beats
    return resample(beats, target_length, axis=1)

# ===============================
# Load model & label encoder
# ===============================

MODEL_PATH = "models/1D_CNN/best_model.h5"
ENCODER_PATH = "models/1D_CNN/label_encoder.json"

model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "r") as f:
    label_map = json.load(f)

classes = label_map["classes"]
inv_label_map = {i: c for i, c in enumerate(classes)}

# ===============================
# Main inference function
# ===============================
def run_ecg_inference(ecg_csv, fs=360, return_attention=False):
    """
    Input:
        ecg_csv: path or file-like object (.csv)
        fs: sampling frequency
        return_attention: bool, optional attention weights
    Output:
        dict with prediction results
    """
    # 1. Load ECG
    df = pd.read_csv(ecg_csv)
    signal = df.iloc[:, 0].values

    # 2. Preprocessing
    filtered = bandpass_filter(signal, fs=fs)
    r_peaks = wp.gqrs_detect(sig=filtered, fs=fs)

    # 3. Extract beats
    beats = extract_beats(filtered, r_peaks)
    if len(beats) == 0:
        return {"error": "No valid beats detected"}

    # 4. Resize to model input length
    beats = resize_beats(beats, target_length=250)

    # 5. Normalize
    beats = normalize_beats(beats)

    # 6. Add channel dimension
    X = beats[..., np.newaxis]  # shape = (num_beats, 250, 1)

    # 7. Predict
    preds = model.predict(X, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    labels = [inv_label_map[int(i)] for i in pred_classes]

    # 8. Aggregate results
    majority_label = Counter(labels).most_common(1)[0][0]
    avg_confidence = float(np.mean(confidences))

    result = {
        "predicted_label": majority_label,
        "confidence": round(avg_confidence * 100, 2),
        "beat_distribution": dict(Counter(labels))
    }

    # Optional attention
    if return_attention:
        # attention model must have output layer accessible
        attention_model = load_model(MODEL_PATH, compile=False)
        attention_weights = attention_model.predict(X[:1])
        result["attention_weights"] = attention_weights.tolist()

    return result
