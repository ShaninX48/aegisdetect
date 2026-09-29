"""Deep-model evaluation - pretrained ViT AI-image detector.
Model: umm-maybe/AI-image-detector (2022, artistic images, ViT, ~348MB).
Limitation (thesis-এ লিখবা): old generators (VQGAN/CLIP era), Midjourney5/SDXL/DALLE-3 unseen,
not a photo-deepfake detector. Use as ONE indicator, not proof.

Usage:
  pip install -r requirements-deep.txt   # CPU torch, one-time ~500MB download
  python deep_eval.py --data-dir ./data_real --out deep_eval_real.json
"""
import argparse
import io
import json
from pathlib import Path

MODEL_ID = "umm-maybe/AI-image-detector"


def load_pipe():
    from transformers import pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", MODEL_ID, device=device)


def ai_prob_from_outputs(outputs) -> float:
    """Map HF labels -> P(AI). Labels are like 'artificial'/'human' (case varies)."""
    total_ai, total_human = 0.0, 0.0
    for o in outputs:
        lab = o["label"].lower()
        s = float(o["score"])
        if any(k in lab for k in ("artific", "ai", "fake", "generat", "synth")):
            total_ai += s
        elif any(k in lab for k in ("human", "real", "authentic", "natural")):
            total_human += s
    if total_ai + total_human == 0:
        # unknown label scheme -> assume first output is AI class
        return float(outputs[0]["score"])
    return total_ai / (total_ai + total_human)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="deep_eval_real.json")
    ap.add_argument("--max-per-class", type=int, default=50)
    args = ap.parse_args()

    from PIL import Image
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    data = Path(args.data_dir)
    real = sorted([p for p in (data / "real").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])[:args.max_per_class]
    ai = sorted([p for p in (data / "ai").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])[:args.max_per_class]
    if not real or not ai:
        raise SystemExit(f"Need images in {data}/real and {data}/ai")

    print(f"Loading {MODEL_ID} (first run downloads ~350MB)...")
    pipe = load_pipe()

    def probs(paths):
        out = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            out.append(ai_prob_from_outputs(pipe(img)))
        return out

    print(f"Scoring {len(real)} real + {len(ai)} ai...")
    pr, pa = probs(real), probs(ai)
    y = [0] * len(pr) + [1] * len(pa)
    proba = pr + pa
    import numpy as np
    proba_np = np.array(proba)
    pred = (proba_np > 0.5).astype(int)
    report = {
        "model": MODEL_ID,
        "n_real": len(real), "n_ai": len(ai),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "3way": {
            "likely_ai": int((proba_np > 0.70).sum()),
            "inconclusive": int(((proba_np >= 0.40) & (proba_np <= 0.70)).sum()),
            "likely_authentic": int((proba_np < 0.40).sum()),
        },
        "samples": [
            {"file": p.name, "true": "real" if i < len(pr) else "ai", "ai_prob": round(float(proba[i]), 4)}
            for i, p in enumerate(real + ai)
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
