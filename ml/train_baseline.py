"""AegisVision baseline - Week 2.
CPU-only, no GPU needed. Frequency + texture features + LogisticRegression.

Usage:
  pip install -r requirements-ml.txt
  python train_baseline.py --demo            # generates demo data, trains, evaluates
  python train_baseline.py --data-dir ./data # expects data/real/*.jpg + data/ai/*.jpg

Output:
  model.pkl, thresholds.json, metrics.json
"""
import argparse
import io
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

SIZE = 128


def extract_features(content: bytes) -> np.ndarray:
    """Spatial + frequency features. Key thesis point: RGB-only vs RGB+Freq ablation."""
    img = Image.open(io.BytesIO(content)).convert("RGB").resize((SIZE, SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)

    # --- spatial ---
    mean = gray.mean()
    std = gray.std()
    # edge density via simple gradient
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    color_var = arr.std(axis=(0, 1)).mean()

    # --- frequency (2D FFT, no scipy needed) ---
    f = np.fft.fft2(gray - gray.mean())
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = mag.sum() + 1e-8
    low = mag[dist < 20].sum() / total       # low-freq energy ratio
    high = mag[dist > 40].sum() / total      # high-freq energy ratio (AI artifacts live here)
    mid = 1.0 - low - high

    return np.array([mean, std, gx, gy, color_var, low, mid, high], dtype=np.float64)


def make_demo_images(out: Path, n_per_class: int = 120):
    """Generate separable demo data so pipeline runs without GenImage download."""
    rng = np.random.default_rng(42)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "ai").mkdir(parents=True, exist_ok=True)
    # real: smooth gradient + soft noise (natural photos ~ low high-freq)
    for i in range(n_per_class):
        base = np.linspace(0, 1, SIZE)[None, :].repeat(SIZE, axis=0)
        base = base * rng.uniform(0.3, 0.9) + rng.uniform(0, 0.3)
        noise = rng.normal(0, 0.03, (SIZE, SIZE))
        g = np.clip(base + noise, 0, 1)
        rgb = np.stack([g, np.roll(g, 5, 1), np.roll(g, -5, 0)], axis=2)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(out / "real" / f"r{i:04d}.png")
    # ai: checkerboard + ringing artifacts (synthetic ~ high high-freq)
    for i in range(n_per_class):
        x = np.arange(SIZE)[:, None] + np.arange(SIZE)[None, :]
        checker = ((x // 4) % 2).astype(float) * 0.35
        ring = np.sin(np.arange(SIZE)[:, None] * 0.6) * np.cos(np.arange(SIZE)[None, :] * 0.6) * 0.15
        g = np.clip(0.5 + checker + ring + rng.normal(0, 0.02, (SIZE, SIZE)), 0, 1)
        rgb = np.stack([g, g * 0.9, g * 1.05 % 1.0], axis=2)
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out / "ai" / f"a{i:04d}.png")
    print(f"Demo data written to {out} ({n_per_class} per class)")


def load_dataset(data_dir: Path):
    X, y = [], []
    for label, sub in [(0, "real"), (1, "ai")]:
        for p in sorted((data_dir / sub).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            X.append(extract_features(p.read_bytes()))
            y.append(label)
    if not X:
        raise SystemExit(f"No images in {data_dir}/real and {data_dir}/ai")
    return np.stack(X), np.array(y)


def apply_threshold(p: float) -> str:
    if p > 0.70:
        return "likely_ai"
    if p < 0.40:
        return "likely_authentic"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data-dir", type=str, default="./data_demo")
    ap.add_argument("--out", type=str, default=".")
    args = ap.parse_args()

    out = Path(args.out)
    data_dir = Path(args.data_dir)
    if args.demo:
        make_demo_images(data_dir)

    X, y = load_dataset(data_dir)
    print(f"Loaded: {len(y)} samples ({(y==0).sum()} real, {(y==1).sum()} ai), features={X.shape[1]}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    base = LogisticRegression(max_iter=2000)
    # calibration = PRD requirement (raw softmax != confidence)
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba > 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        "n_test": len(yte),
    }
    # 3-way distribution with PRD thresholds
    cats = [apply_threshold(p) for p in proba]
    metrics["3way"] = {c: cats.count(c) for c in ["likely_ai", "inconclusive", "likely_authentic"]}
    print(json.dumps(metrics, indent=2))

    with open(out / "model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "thresholds.json", "w") as f:
        json.dump({"likely_ai": 0.70, "likely_authentic": 0.40, "model": "AegisVision v1.0-baseline"}, f, indent=2)
    print(f"Saved model.pkl, metrics.json to {out.resolve()}")
    print("Next: replace classify_dummy() in backend with aegis_vision.predict()")


if __name__ == "__main__":
    main()
