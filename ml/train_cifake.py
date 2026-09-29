"""CIFAKE fine-tune - thesis-grade numbers on CPU.
Dataset: yanbax/CIFAKE_autotrain_compatible (100k, SD-v1.4 fake + CIFAR-10 real).
Paper baseline: 92.98% CNN accuracy (Bird & Lotfi 2023) - cite it.

Usage:
  pip install datasets   # + torch already installed
  python train_cifake.py --max-train 4000 --max-test 1000 --epochs 5
  python train_cifake.py --max-train 20000 --epochs 10   # overnight, better numbers

Output: cifake_cnn.pt, cifake_metrics.json
Then: python eval_real.py --data-dir ./data_real (compare in-dist vs unseen)
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMG = 32


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(1)


def to_tensor(img) -> np.ndarray:
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    img = img.convert("RGB").resize((IMG, IMG))
    return np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0


def label_of(row) -> int:
    # Defactify: Label_A (Task A: 0=real, 1=fake), CIFAKE: label/labels, generic fallbacks
    low = {k.lower(): v for k, v in row.items()}
    for k in ("label_a", "label", "labels", "class", "target", "category", "label_b"):
        if k in low:
            v = low[k]
            if isinstance(v, str):
                vl = v.lower()
                if vl in ("fake", "ai", "synthetic", "generated", "1", "artificial"):
                    return 1
                if vl in ("real", "human", "authentic", "natural", "0"):
                    return 0
                try:
                    return int(float(vl))  # '0'/'1' strings
                except ValueError:
                    continue
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    raise KeyError(f"No label column in {list(row.keys())}")


def img_of(row):
    low = {k.lower(): v for k, v in row.items()}
    for k in ("image", "img", "pixel_values", "file"):
        if k in low and low[k] is not None:
            return low[k]
    raise KeyError(f"No image column in {list(row.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yanbax/CIFAKE_autotrain_compatible")
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=None,
                    help="Local folder with real/fake subfolders (e.g. Kaggle CIFAKE train/). Overrides --dataset.")
    ap.add_argument("--streaming", action="store_true",
                    help="Stream HF dataset (no full download). Use for large sets like Defactify.")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from datasets import load_dataset
    if args.data_dir:
        from PIL import Image as PILImage
        base = Path(args.data_dir)
        # support CIFAKE-Kaggle layout (train/REAL, train/FAKE) or flat (real/ai, real/fake)
        cand = [(base / "train" / "REAL", 0), (base / "train" / "FAKE", 1),
                (base / "train" / "real", 0), (base / "train" / "synthetic", 1),
                (base / "REAL", 0), (base / "FAKE", 1),
                (base / "real", 0), (base / "fake", 1), (base / "ai", 1),
                (base / "synthetic", 1)]
        files = []
        for folder, lab in cand:
            if folder.exists():
                for p in sorted(folder.glob("*")):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        files.append((p, lab))
        if not files:
            raise SystemExit(f"No images under {base} (tried train/REAL, train/FAKE, real/fake, ai)")
        random.shuffle(files)
        imgs, labs = [], []
        for p, lab in files[:args.max_train + args.max_test]:
            try:
                imgs.append(to_tensor(PILImage.open(p)))
                labs.append(lab)
            except Exception:
                continue
        Xa = torch.tensor(np.stack(imgs))
        ya = torch.tensor(np.array(labs), dtype=torch.float32)
        n_te = min(args.max_test, len(Xa) // 5)
        Xtr, ytr, Xte, yte = Xa[:-n_te], ya[:-n_te], Xa[-n_te:], ya[-n_te:]
        print(f"Local: {len(Xa)} images from {base} ({int((ya==0).sum())} real, {int((ya==1).sum())} fake)")
        print(f"Columns: [local files]")
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
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
                n += len(xb)
            print(f"epoch {ep+1}/{args.epochs} loss {tot/n:.4f}", flush=True)
        net.eval()
        with torch.no_grad():
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            proba = torch.sigmoid(net(Xte.to(dev))).cpu().numpy()
            pred = (proba > 0.5).astype(int)
            y = yte.numpy().astype(int)
            m = {
                "dataset": f"local:{base}", "n_train": len(Xtr), "n_test": len(Xte), "epochs": args.epochs,
                "accuracy": round(float(accuracy_score(y, pred)), 4),
                "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y, proba)) if len(set(y)) > 1 else 0.0, 4),
            }
        print(json.dumps(m, indent=2))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), out / "cifake_cnn.pt")
        (out / "cifake_metrics.json").write_text(json.dumps(m, indent=2))
        print(f"Saved cifake_cnn.pt to {out.resolve()}")
        return
    print(f"Loading {args.dataset}...")
    stream = getattr(args, "streaming", False)
    if stream:
        ds = load_dataset(args.dataset, streaming=True)
        names = list(ds.keys())
        tr_key = "train" if "train" in names else names[0]
        te_key = "test" if "test" in names else ("validation" if "validation" in names else names[0])
        print(f"Streaming splits: {names} -> train={tr_key}, test={te_key}")

        def collect_stream(split, n):
            X, y = [], []
            it = split.shuffle(seed=args.seed, buffer_size=2000) if hasattr(split, "shuffle") else split
            for row in it:
                try:
                    X.append(to_tensor(img_of(row)))
                    y.append(label_of(row))
                except Exception as e:
                    if len(X) == 0:
                        print("First-row keys:", list(row.keys()))
                    continue
                if len(X) >= n:
                    break
            return torch.tensor(np.stack(X)), torch.tensor(np.array(y), dtype=torch.float32)

        Xtr, ytr = collect_stream(ds[tr_key], args.max_train)
        Xte, yte = collect_stream(ds[te_key], args.max_test)
    else:
        ds = load_dataset(args.dataset)
        # pick splits: prefer train/test, else split train
        if "test" in ds:
            tr, te = ds["train"], ds["test"]
        elif "validation" in ds:
            tr, te = ds["train"], ds["validation"]
        else:
            s = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
            tr, te = s["train"], s["test"]
        print(f"Columns: {tr.column_names}")

        def collect(split, n):
            idx = random.sample(range(len(split)), min(n, len(split)))
            X, y = [], []
            for i in idx:
                row = split[i]
                try:
                    X.append(to_tensor(img_of(row)))
                    y.append(label_of(row))
                except Exception:
                    continue
            return torch.tensor(np.stack(X)), torch.tensor(np.array(y), dtype=torch.float32)

        Xtr, ytr = collect(tr, args.max_train)
        Xte, yte = collect(te, args.max_test)
    print(f"Train {tuple(Xtr.shape)} pos-rate {ytr.mean():.2f} | Test {tuple(Xte.shape)}")

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
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        print(f"epoch {ep+1}/{args.epochs} loss {tot/n:.4f}", flush=True)

    net.eval()
    with torch.no_grad():
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        proba = torch.sigmoid(net(Xte.to(dev))).cpu().numpy()
        pred = (proba > 0.5).astype(int)
        y = yte.numpy().astype(int)
        m = {
            "dataset": args.dataset, "n_train": len(Xtr), "n_test": len(Xte), "epochs": args.epochs,
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        }
    print(json.dumps(m, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "cifake_cnn.pt")
    (out / "cifake_metrics.json").write_text(json.dumps(m, indent=2))
    print(f"Saved cifake_cnn.pt to {out.resolve()}")
    print("Paper baseline 92.98% (full 100k, tuned CNN). Your subset will be lower - report both.")
    (out / "cifake_metrics.json").write_text(json.dumps(m, indent=2))
    print(f"Saved cifake_cnn.pt to {out.resolve()}")
    print("Paper baseline 92.98% (full 100k, tuned CNN). Your subset will be lower - report both.")
    print("Next: test this model on ./data_real as UNSEEN set for generalization gap.")


if __name__ == "__main__":
    main()
