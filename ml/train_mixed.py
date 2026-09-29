"""Mixed training: SynthCIFAR + data_real(70% augmented) -> test on data_real 30% held-out.
Leakage-safe: held-out IDs never seen in train (PRD-12). Augmentations simulate WhatsApp compression.

Usage: python train_mixed.py --synth-train 3000 --epochs 8 --out ./mixed_out
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
from torch.utils.data import DataLoader, TensorDataset

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


def to_tensor(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG, IMG))
    return np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0


def augment(pil: Image.Image, rng: np.random.Generator):
    """Original + WhatsApp-like JPEG + downscale + flip."""
    outs = [pil]
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=60)
    outs.append(Image.open(buf))
    w, h = pil.size
    small = pil.resize((max(8, w // 2), max(8, h // 2))).resize((w, h), Image.BILINEAR)
    outs.append(small)
    outs.append(pil.transpose(Image.FLIP_LEFT_RIGHT))
    return outs


def gather(folder: Path, exts=(".jpg", ".jpeg", ".png", ".webp")):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", default="./synthcifar")
    ap.add_argument("--real-dir", default="./data_real")
    ap.add_argument("--synth-train", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="./mixed_out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real-repeat", type=int, default=8,
                    help="Repeat augmented real-world samples to balance synth:real ratio.")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- 1. SynthCIFAR subset (balanced) ---
    syn = Path(args.synth_dir)
    r_all = gather(syn / "train" / "real")
    s_all = gather(syn / "train" / "synthetic")
    n_half = args.synth_train // 2
    X, y = [], []
    for p in random.sample(r_all, min(n_half, len(r_all))):
        X.append(to_tensor(Image.open(p))); y.append(0)
    for p in random.sample(s_all, min(n_half, len(s_all))):
        X.append(to_tensor(Image.open(p))); y.append(1)
    print(f"Synth part: {len(X)} ({y.count(0)} real, {y.count(1)} synth)")

    # --- 2. data_real stratified 70/30 split (by file, leakage-safe) ---
    base = Path(args.real_dir)
    real = sorted((base / "real").glob("*"))
    ai = sorted((base / "ai").glob("*"))
    real = [p for p in real if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    ai = [p for p in ai if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    random.shuffle(real); random.shuffle(ai)
    r_tr, r_te = real[:int(0.7 * len(real))], real[int(0.7 * len(real)):]
    a_tr, a_te = ai[:int(0.7 * len(ai))], ai[int(0.7 * len(ai)):]
    print(f"data_real split: train {len(r_tr)}R+{len(a_tr)}A, held-out {len(r_te)}R+{len(a_te)}A")
    held_paths = r_te + a_te
    held_y = [0] * len(r_te) + [1] * len(a_te)
    real_aug = []
    for p in r_tr:
        for aug in augment(Image.open(p).convert("RGB"), rng):
            real_aug.append((to_tensor(aug), 0))
    for p in a_tr:
        for aug in augment(Image.open(p).convert("RGB"), rng):
            real_aug.append((to_tensor(aug), 1))
    # rebalance: repeat real-world samples so synth doesn't drown them
    X_real = [t for t, _ in real_aug] * args.real_repeat
    y_real = [l for _, l in real_aug] * args.real_repeat
    for t, l in zip(X_real, y_real):
        X.append(t); y.append(l)
    print(f"Real-world samples after x{args.real_repeat} repeat: {len(X_real)} (synth:{len(X)-len(X_real)} real:{len(X_real)})")
    print(f"Total train: {len(X)} (synth + augmented real-world)")

    Xtr = torch.tensor(np.stack(X))
    ytr = torch.tensor(np.array(y), dtype=torch.float32)
    dev = torch.device("cpu")
    net = TinyCNN().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, shuffle=True)
    for ep in range(args.epochs):
        net.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(net(xb.to(dev)), yb.to(dev))
            loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        print(f"epoch {ep+1}/{args.epochs} loss {tot/n:.4f}", flush=True)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    net.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "mixed_cnn.pt")

    def score(paths, labels, name):
        with torch.no_grad():
            Xt = torch.stack([torch.tensor(to_tensor(Image.open(p).convert("RGB"))) for p in paths])
            proba = torch.sigmoid(net(Xt.to(dev))).cpu().numpy()
        pred = (proba > 0.5).astype(int)
        m = {
            "n": len(paths),
            "accuracy": round(float(accuracy_score(labels, pred)), 4),
            "precision": round(float(precision_score(labels, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(labels, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(labels, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(labels, proba)) if len(set(labels)) > 1 else 0.0, 4),
        }
        print(f"{name}: {json.dumps(m)}")
        return m

    # held-out real-world test
    held = score(held_paths, held_y, "held-out(data_real 30%)")
    # retention: synthcifar test subset
    te_r = random.sample(gather(syn / "test" / "real"), 250)
    te_s = random.sample(gather(syn / "test" / "synthetic"), 250)
    reten = score(te_r + te_s, [0] * 250 + [1] * 250, "retention(synth test 500)")
    rep = {"held_out": held, "retention": reten,
           "train_n": len(Xtr), "epochs": args.epochs, "seed": args.seed,
           "held_out_ids": [p.name for p in held_paths]}
    (out / "mixed_metrics.json").write_text(json.dumps(rep, indent=2))
    print(f"Saved mixed_cnn.pt to {out.resolve()}")


if __name__ == "__main__":
    main()
