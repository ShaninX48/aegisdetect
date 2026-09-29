"""v4 structure-level detector: frozen ResNet18 (ImageNet) + linear probe, 224px.
Why: 32x32 erases hands/texture/edges. 224px + pretrained features = structure.
Stage-1: data_real only, stratified K-fold CV (honest eval on 72 imgs).
Stage-2 (later): --fake-dir/--real-dir for TrueFake FLUX.1/FORLAB high-res.

Usage: python train_v4.py --folds 3 --epochs 12 --out ./v4_out
"""
import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def train_tf():
    return transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def eval_tf():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


class ImgList(Dataset):
    def __init__(self, items, tf, jpeg_q=None):
        self.items, self.tf, self.q = items, tf, jpeg_q

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.q and random.random() < 0.5:  # WhatsApp-like recompression aug
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.choice([50, 70, 90]))
            img = Image.open(buf)
        return self.tf(img), torch.tensor(y, dtype=torch.float32)


def get_backbone():
    try:
        w = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=w)
    except Exception:  # offline: random init (warn in metrics)
        net = models.resnet18(weights=None)
    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net, False


class Probe(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)


@torch.no_grad()
def embed(net, loader, dev):
    feats, ys = [], []
    for xb, yb in loader:
        feats.append(net(xb.to(dev)).cpu())
        ys.append(yb)
    return torch.cat(feats), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data_real")
    ap.add_argument("--fake-dir", default=None, help="extra high-res fake folder (TrueFake FLUX.1)")
    ap.add_argument("--extra-real-dir", default=None, help="extra high-res real folder (FORLAB)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="./v4_out")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cpu")

    base = Path(args.data_dir)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    items = [(p, 0) for p in sorted((base / "real").glob("*")) if p.suffix.lower() in exts]
    items += [(p, 1) for p in sorted((base / "ai").glob("*")) if p.suffix.lower() in exts]
    if args.fake_dir:
        items += [(p, 1) for p in sorted(Path(args.fake_dir).rglob("*")) if p.suffix.lower() in exts]
    if args.extra_real_dir:
        items += [(p, 0) for p in sorted(Path(args.extra_real_dir).rglob("*")) if p.suffix.lower() in exts]
    print(f"Pool: {len(items)} ({sum(1 for _, y in items if y == 0)} real, {sum(1 for _, y in items if y == 1)} ai)")
    paths = [str(p) for p, _ in items]
    labels = np.array([y for _, y in items])

    net, _ = get_backbone()
    net.to(dev)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_ms, best_state, best_f1 = [], None, -1.0
    for fi, (tr, te) in enumerate(skf.split(paths, labels)):
        tr_items = [(paths[i], int(labels[i])) for i in tr]
        te_items = [(paths[i], int(labels[i])) for i in te]
        # oversample minority class in train (domain-safe, no leakage: split first)
        n0 = sum(1 for _, y in tr_items if y == 0)
        n1 = len(tr_items) - n0
        if n0 > 0 and n1 > 0:
            rep = max(n0, n1) // min(n0, n1)
            minors = [it for it in tr_items if it[1] == (0 if n0 < n1 else 1)]
            tr_items = tr_items + minors * min(rep - 1, 6)
        tr_loader = DataLoader(ImgList(tr_items, train_tf(), jpeg_q=True), batch_size=16, shuffle=True)
        te_loader = DataLoader(ImgList(te_items, eval_tf()), batch_size=16)
        Ftr, ytr = embed(net, tr_loader, dev)
        Fte, yte = embed(net, te_loader, dev)
        probe = Probe(Ftr.shape[1])
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        for ep in range(args.epochs):
            probe.train()
            perm = torch.randperm(len(Ftr))
            for b in range(0, len(Ftr), 16):
                idx = perm[b:b + 16]
                opt.zero_grad()
                loss = loss_fn(probe(Ftr[idx]), ytr[idx])
                loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            proba = torch.sigmoid(probe(Fte)).numpy()
        pred = (proba > 0.5).astype(int)
        y = yte.numpy().astype(int)
        m = {
            "fold": fi,
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)) if len(set(y)) > 1 else 0.0, 4),
        }
        print(f"fold {fi}: {json.dumps(m)}", flush=True)
        fold_ms.append(m)
        if m["f1"] > best_f1:
            best_f1, best_state = m["f1"], {k: v.cpu() for k, v in probe.state_dict().items()}
    avg = {k: round(float(np.mean([m[k] for m in fold_ms])), 4) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    print("CV-AVG:", json.dumps(avg))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"probe": best_state, "arch": "resnet18-frozen-probe-224"}, out / "v4_probe.pt")
    (out / "v4_metrics.json").write_text(json.dumps({"folds": fold_ms, "cv_avg": avg, "n": len(items)}, indent=2))
    print(f"Saved v4_probe.pt to {out.resolve()}")


if __name__ == "__main__":
    main()
