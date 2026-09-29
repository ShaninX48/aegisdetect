"""Inference wrapper - drop-in replacement for classify_dummy() in backend.
Copy this file to backend/app/services/aegis_vision.py, put model.pkl next to it.
"""
import pickle
from pathlib import Path

import numpy as np

_MODEL = None
_MODEL_PATH = Path(__file__).parent / "model.pkl"
SIZE = 128


def _features_from_bytes(content: bytes) -> np.ndarray:
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(content)).convert("RGB").resize((SIZE, SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    mean, std = float(gray.mean()), float(gray.std())
    gx = float(np.abs(np.diff(gray, axis=1)).mean())
    gy = float(np.abs(np.diff(gray, axis=0)).mean())
    color_var = float(arr.std(axis=(0, 1)).mean())
    f = np.fft.fft2(gray - gray.mean())
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = float(mag.sum()) + 1e-8
    low = float(mag[dist < 20].sum() / total)
    high = float(mag[dist > 40].sum() / total)
    mid = 1.0 - low - high
    return np.array([[mean, std, gx, gy, color_var, low, mid, high]])


def load_model():
    global _MODEL
    if _MODEL is None:
        with open(_MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
    return _MODEL


def predict(content: bytes):
    """Returns (result, ai_prob, indicators). Same signature as classify_dummy."""
    try:
        clf = load_model()
        ai_prob = round(float(clf.predict_proba(_features_from_bytes(content))[0, 1]), 4)
    except FileNotFoundError:
        # model.pkl missing -> fall back so backend never crashes
        return "inconclusive", 0.5, [
            {"code": "no_model", "label": "model.pkl not found, run train_baseline.py --demo", "score": 0.5}
        ]
    if ai_prob > 0.70:
        result = "likely_ai"
    elif ai_prob < 0.40:
        result = "likely_authentic"
    else:
        result = "inconclusive"
    indicators = [
        {"code": "freq_high_energy", "label": "High-frequency energy analysis", "score": ai_prob},
        {"code": "texture_model", "label": "AegisVision v1.0-baseline (calibrated LR)", "score": ai_prob},
    ]
    return result, ai_prob, indicators
