"""Backend plug-in for voice baseline (self-contained, WAV only).
Install: copy this file -> backend/app/services/voice_model.py + voice_model.pkl beside it.
MP3/M4A: backend returns 'needs WAV' indicator until ffmpeg decoding lands (Phase-2+).
"""
import io
import pickle
import struct
import wave
from pathlib import Path

import numpy as np

_MODEL = None
_MODEL_PATH = Path(__file__).parent / "voice_model.pkl"
MODEL_VERSION = "AegisVoice v1.0-baseline"
SR, FRAME = 16000, 1024


def _read_mono(content: bytes):
    with wave.open(io.BytesIO(content), "rb") as w:
        n, sw, ch, sr = w.getnframes(), w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    x = np.array(struct.unpack("<" + {1: "b", 2: "h", 4: "i"}[sw] * (len(raw) // sw), raw), dtype=np.float64)
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR and len(x) > 0:
        idx = np.clip((np.arange(int(len(x) * SR / sr)) * sr / SR).astype(int), 0, len(x) - 1)
        x = x[idx]
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def _features(content: bytes) -> np.ndarray:
    x = _read_mono(content)
    if len(x) < FRAME * 4:
        x = np.pad(x, (0, FRAME * 4 - len(x)))
    cents, bws, rolls, zcrs, engs = [], [], [], [], []
    for i in range(len(x) // FRAME):
        fr = x[i * FRAME:(i + 1) * FRAME] * np.hanning(FRAME)
        mag = np.abs(np.fft.rfft(fr))
        freqs = np.fft.rfftfreq(FRAME, 1 / SR)
        tot = mag.sum() + 1e-9
        cent = (freqs * mag).sum() / tot
        bw = np.sqrt(((freqs - cent) ** 2 * mag).sum() / tot)
        rolls.append(freqs[np.searchsorted(np.cumsum(mag), 0.85 * tot).clip(0, len(freqs) - 1)])
        cents.append(cent); bws.append(bw)
        zcrs.append(((fr[:-1] * fr[1:]) < 0).mean())
        engs.append((fr ** 2).mean())
    cents, bws, rolls, zcrs, engs = map(np.array, (cents, bws, rolls, zcrs, engs))
    loge = np.log10(engs + 1e-9)
    return np.array([[cents.mean(), cents.std(), bws.mean(), rolls.mean(),
                      zcrs.mean(), zcrs.std(), loge.mean(), loge.std(),
                      np.abs(np.diff(loge)).mean()]])


def _load():
    global _MODEL
    if _MODEL is None:
        with open(_MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
    return _MODEL


def predict_wav(content: bytes):
    try:
        clf = _load()
    except FileNotFoundError:
        return "inconclusive", 0.5, [
            {"code": "no_voice_model", "label": "voice_model.pkl missing, run voice_baseline.py --demo", "score": 0.5}
        ]
    ai_prob = round(float(clf.predict_proba(_features(content))[0, 1]), 4)
    result = "likely_ai" if ai_prob > 0.70 else ("likely_authentic" if ai_prob < 0.40 else "inconclusive")
    return result, ai_prob, [
        {"code": "voice_spectral", "label": "Spectral/prosody analysis (calibrated LR)", "score": ai_prob}
    ]
