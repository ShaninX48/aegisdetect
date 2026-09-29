"""Real-data evaluation - PRD section 13/14/28.
Evaluates plugged model on ANY folder: data/real/*.jpg + data/ai/*.jpg
plus robustness: JPEG q70, resize 0.5x, Gaussian noise.

Usage:
  python eval_real.py --data-dir ./data_demo        # demo check
  python eval_real.py --data-dir ./data_real        # your real photos + AI images
  python eval_real.py --data-dir ./data_real --model ../backend/app/services/model.pkl

Output: eval_report.json (accuracy, precision, recall, F1, ROC-AUC, FPR, 3way, robustness drop)
"""
import argparse
import io
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SIZE = 128


def extract_features(content: bytes) -> np.ndarray:
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
    return np.array([mean, std, gx, gy, color_var, low, 1.0 - low - high, high])


def jpeg_bytes(img: Image.Image, q: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    return buf.getvalue()


def eval_split(clf, paths, labels):
    X = np.stack([extract_features(p.read_bytes()) for p in paths])
    y = np.array(labels)
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    # FPR on real = FP / (FP+TN)
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "n": len(y),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)) if len(set(y)) > 1 else 0.0, 4),
        "fpr_on_real": round(fp / max(1, fp + tn), 4),
        "3way": {
            "likely_ai": int((proba > 0.70).sum()),
            "inconclusive": int(((proba >= 0.40) & (proba <= 0.70)).sum()),
            "likely_authentic": int((proba < 0.40).sum()),
        },
    }, proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="../backend/app/services/model.pkl")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    data = Path(args.data_dir)
    with open(args.model, "rb") as f:
        clf = pickle.load(f)

    real = sorted([p for p in (data / "real").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]) if (data / "real").exists() else []
    ai = sorted([p for p in (data / "ai").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]) if (data / "ai").exists() else []
    if not real or not ai:
        raise SystemExit(f"Need {data}/real/* and {data}/ai/* with images. Found real={len(real)}, ai={len(ai)}")

    paths, labels = real + ai, [0] * len(real) + [1] * len(ai)
    clean, _ = eval_split(clf, paths, labels)

    # --- robustness: transform in-memory, re-evaluate ---
    rng = np.random.default_rng(0)
    trans = {}
    for name in ["jpeg_q70", "resize_half", "gauss_noise"]:
        tpaths = []
        tmpdir = Path("_eval_tmp") / name
        tmpdir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            img = Image.open(p).convert("RGB")
            if name == "jpeg_q70":
                content = jpeg_bytes(img, 70)
            elif name == "resize_half":
                small = img.resize((max(16, img.width // 2), max(16, img.height // 2)))
                buf = io.BytesIO()
                small.resize((img.width, img.height)).save(buf, "PNG")
                content = buf.getvalue()
            else:
                arr = (np.asarray(img).astype(float) + rng.normal(0, 8, (img.height, img.width, 1))).clip(0, 255).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, "PNG")
                content = buf.getvalue()
            tp = tmpdir / p.name
            tp.write_bytes(content)
            tpaths.append(tp)
        m, _ = eval_split(clf, tpaths, labels)
        trans[name] = {"f1": m["f1"], "accuracy": m["accuracy"], "drop_vs_clean_f1": round(clean["f1"] - m["f1"], 4)}

    report = {"data_dir": str(data), "model": args.model, "clean": clean, "robustness": trans}
    target = {"in_dist_F1>=0.90": clean["f1"] >= 0.90, "FPR_on_real<0.05": clean["fpr_on_real"] < 0.05,
              "compressed_drop<0.10": abs(trans["jpeg_q70"]["drop_vs_clean_f1"]) < 0.10}
    report["PRD_targets"] = target
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
