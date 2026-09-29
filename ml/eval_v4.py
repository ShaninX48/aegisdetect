"""Eval saved v4 probe on arbitrary real/ai folders + 3way.
Usage: python eval_v4.py --model ./v4_stage2_out/v4_probe.pt --test ./defactify_cap/test --unseen ./data_real
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(SIZE),
                         transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class L(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return TF(Image.open(p).convert("RGB")), torch.tensor(y, dtype=torch.float32)


def load_dir(base: Path):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    items = [(p, 0) for p in sorted((base / "real").glob("*")) if p.suffix.lower() in exts]
    items += [(p, 1) for p in sorted((base / "ai").glob("*")) if p.suffix.lower() in exts]
    return items


def run(net, probe, items, name, dev):
    loader = DataLoader(L(items), batch_size=16)
    net.eval(); probe.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps.append(torch.sigmoid(probe(net(xb.to(dev)))).cpu().numpy())
            ys.append(yb.numpy())
    proba, y = np.concatenate(ps), np.concatenate(ys).astype(int)
    pred = (proba > 0.5).astype(int)
    m = {"n": len(y),
         "accuracy": round(float(accuracy_score(y, pred)), 4),
         "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
         "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
         "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
         "roc_auc": round(float(roc_auc_score(y, proba)), 4),
         "3way": {"likely_ai": int((proba > 0.70).sum()),
                  "inconclusive": int(((proba >= 0.40) & (proba <= 0.70)).sum()),
                  "likely_authentic": int((proba < 0.40).sum())}}
    print(f"{name}: {json.dumps(m)}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--unseen", default=None)
    args = ap.parse_args()
    dev = torch.device("cpu")
    try:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        net = models.resnet18(weights=None)
    net.fc = nn.Identity()
    net.to(dev).eval()
    ckpt = torch.load(args.model, map_location="cpu")
    probe = nn.Linear(512, 1)
    probe.load_state_dict({k.split(".", 1)[1] if "." in k else k: v for k, v in ckpt["probe"].items()})
    probe.to(dev).eval()
    rep = {}
    if args.test:
        rep["heldout_modern"] = run(net, probe, load_dir(Path(args.test)), "heldout-modern", dev)
    if args.unseen:
        rep["unseen_phones"] = run(net, probe, load_dir(Path(args.unseen)), "unseen-phones", dev)
    Path("v4_eval_report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
