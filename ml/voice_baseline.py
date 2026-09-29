"""AegisVoice baseline - CPU-only, no librosa needed.
Features: spectral centroid/bandwidth/rolloff, ZCR, energy dynamics (numpy FFT only).
Target data: ASVspoof2019 LA (later). Now: --demo synthetic check + your own folders.

Usage:
  python voice_baseline.py --demo                        # generates demo wavs, trains
  python voice_baseline.py --data-dir ./voice_data       # voice_data/real/*.wav + voice_data/ai/*.wav

Output: voice_model.pkl, voice_metrics.json
"""
import argparse
import io
import json
import pickle
import struct
import wave
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

SR = 16000
FRAME = 1024


def read_wav_mono(content: bytes):
    with wave.open(io.BytesIO(content), "rb") as w:
        n, sw, ch, sr = w.getnframes(), w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    fmt = {1: "b", 2: "h", 4: "i"}[sw]
    x = np.array(struct.unpack("<" + fmt * (len(raw) // sw), raw), dtype=np.float64)
    x = x.reshape(-1, ch).mean(axis=1)
    # resample naive to 16k if needed (decimate/nearest)
    if sr != SR and len(x) > 0:
        idx = (np.arange(int(len(x) * SR / sr)) * sr / SR).astype(int)
        idx = np.clip(idx, 0, len(x) - 1)
        x = x[idx]
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak
    return x


def extract_voice_features(content: bytes) -> np.ndarray:
    x = read_wav_mono(content)
    if len(x) < FRAME * 4:
        x = np.pad(x, (0, FRAME * 4 - len(x)))
    n_frames = len(x) // FRAME
    cents, bws, rolls, zcrs, engs = [], [], [], [], []
    for i in range(n_frames):
        fr = x[i * FRAME:(i + 1) * FRAME] * np.hanning(FRAME)
        mag = np.abs(np.fft.rfft(fr))
        freqs = np.fft.rfftfreq(FRAME, 1 / SR)
        tot = mag.sum() + 1e-9
        cent = (freqs * mag).sum() / tot
        bw = np.sqrt(((freqs - cent) ** 2 * mag).sum() / tot)
        roll = freqs[np.searchsorted(np.cumsum(mag), 0.85 * tot).clip(0, len(freqs) - 1)]
        zcr = ((fr[:-1] * fr[1:]) < 0).mean()
        eng = (fr ** 2).mean()
        cents.append(cent); bws.append(bw); rolls.append(roll); zcrs.append(zcr); engs.append(eng)
    cents, bws, rolls, zcrs, engs = map(np.array, (cents, bws, rolls, zcrs, engs))
    loge = np.log10(engs + 1e-9)
    feats = [
        cents.mean(), cents.std(), bws.mean(), rolls.mean(),
        zcrs.mean(), zcrs.std(), loge.mean(), loge.std(),
        np.abs(np.diff(loge)).mean(),  # prosody/energy dynamics (flat in robotic TTS)
    ]
    return np.array(feats)


def synth_wav(path: Path, kind: str, rng: np.random.Generator, i: int):
    dur, t = 2.0, np.arange(int(SR * 2.0)) / SR
    if kind == "real":
        # human-like: vibrato + varying formants + breath noise
        f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t + i) + rng.normal(0, 3, len(t))
        sig = (np.sin(2 * np.pi * np.cumsum(f0) / SR)
               + 0.4 * np.sin(2 * np.pi * np.cumsum(2 * f0) / SR + 1)
               + 0.2 * np.sin(2 * np.pi * np.cumsum(3 * f0) / SR + 2))
        sig += rng.normal(0, 0.05, len(t))
        sig *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * t)  # syllable-like swells
    else:
        # robotic TTS-like: flat pitch, frame-rate ripple, metallic ring
        f0 = 150.0
        sig = np.sign(np.sin(2 * np.pi * f0 * t)) * 0.5 + 0.3 * np.sin(2 * np.pi * f0 * t)
        sig += 0.15 * np.sin(2 * np.pi * 4000 * t)  # vocoder-ish whistle
        sig *= 0.9  # flat energy
    sig = (sig / (np.abs(sig).max() + 1e-9) * 30000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(sig.tobytes())


def make_demo(out: Path, n: int = 100):
    rng = np.random.default_rng(0)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "ai").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        synth_wav(out / "real" / f"r{i:04d}.wav", "real", rng, i)
        synth_wav(out / "ai" / f"a{i:04d}.wav", "ai", rng, i)
    print(f"Demo voice data: {n} per class in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data-dir", default="./voice_demo")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out = Path(args.out)
    data = Path(args.data_dir)
    if args.demo:
        make_demo(data)

    paths = sorted((data / "real").glob("*.wav")) + sorted((data / "ai").glob("*.wav"))
    labels = [0] * len(list((data / "real").glob("*.wav"))) + [1] * len(list((data / "ai").glob("*.wav")))
    X = np.stack([extract_voice_features(p.read_bytes()) for p in paths])
    y = np.array(labels)
    print(f"Loaded {len(y)} clips ({(y==0).sum()} real, {(y==1).sum()} ai), features={X.shape[1]}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = CalibratedClassifierCV(LogisticRegression(max_iter=2000), cv=3)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba > 0.5).astype(int)
    m = {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        "n_test": len(yte),
    }
    print(json.dumps(m, indent=2))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "voice_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    (out / "voice_metrics.json").write_text(json.dumps(m, indent=2))
    print(f"Saved voice_model.pkl to {out.resolve()}")


if __name__ == "__main__":
    main()
