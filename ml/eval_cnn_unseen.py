"""Unseen test: SynthCIFAR-trained TinyCNN on data_real (phone photos + modern AI).
Usage: python eval_cnn_unseen.py --model ./synthcifar_out/cifake_cnn.pt --data-dir ./data_real
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG = 32


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(1)


def load_img(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB").resize((IMG, IMG))
    return torch.tensor(np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./synthcifar_out/cifake_cnn.pt")
    ap.add_argument("--data-dir", default="./data_real")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    net = TinyCNN()
    net.load_state_dict(torch.load(args.model, map_location="cpu"))
    net.eval()

    base = Path(args.data_dir)
    real = sorted([p for p in (base / "real").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    ai_dirs = [base / d for d in ("ai", "fake", "synthetic") if (base / d).exists()]
    ai = sorted([p for d in ai_dirs for p in d.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    paths, y = real + ai, [0] * len(real) + [1] * len(ai)
    print(f"Unseen: {len(real)} real + {len(ai)} ai = {len(paths)}")

    with torch.no_grad():
        X = torch.stack([load_img(p) for p in paths])
        proba = torch.sigmoid(net(X)).numpy()
    pred = (proba > 0.5).astype(int)
    r = {
        "model": args.model, "unseen": str(base),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "3way": {
            "likely_ai": int((proba > 0.70).sum()),
            "inconclusive": int(((proba >= 0.40) & (proba <= 0.70)).sum()),
            "likely_authentic": int((proba < 0.40).sum()),
        },
    }
    print(json.dumps(r, indent=2))
    Path("cnn_unseen_report.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
