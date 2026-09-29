"""Fetch capped Defactify subset to disk (resume-safe), then exit.
Saves JPEGs (q88) to keep it small: out/pool/{real,ai} + out/test/{real,ai}.
Usage: python fetch_defactify.py --cap-train 1200 --cap-test 250 --out ./defactify_cap
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Rajarshi-Roy-research/Defactify_Image_Dataset")
    ap.add_argument("--cap-train", type=int, default=1200)
    ap.add_argument("--cap-test", type=int, default=250)
    ap.add_argument("--out", default="./defactify_cap")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    from datasets import load_dataset
    from PIL import Image

    out = Path(args.out)
    for sub in ("pool/real", "pool/ai", "test/real", "test/ai"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    state_p = out / "fetch_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"train_saved": 0, "test_saved": 0}

    ds = load_dataset(args.dataset, streaming=True)
    # balance classes while streaming: alternate real/fake targets
    targets = {"train": args.cap_train, "test": args.cap_test}
    for split, cap in targets.items():
        have = state["train_saved"] if split == "train" else state["test_saved"]
        if have >= cap:
            print(f"{split}: already have {have}, skip")
            continue
        need_real = need_fake = (cap - have) // 2
        got_real = got_fake = 0
        dest = "pool" if split == "train" else "test"
        print(f"{split}: streaming until +{need_real} real +{need_fake} fake ...", flush=True)
        it = ds[split].shuffle(seed=args.seed, buffer_size=3000)
        for row in it:
            if got_real >= need_real and got_fake >= need_fake:
                break
            try:
                lab = str(row.get("Label_A", row.get("label_a", ""))).strip()
                is_fake = lab == "1"
                if is_fake and got_fake >= need_fake:
                    continue
                if not is_fake and got_real >= need_real:
                    continue
                img = row["Image"] if "Image" in row else row["image"]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                img = img.convert("RGB")
                img.thumbnail((768, 768), Image.BILINEAR)  # cap size, keep structure
                name = f"{split}_{have + got_real + got_fake:05d}.jpg"
                img.save(out / dest / ("ai" if is_fake else "real") / name, "JPEG", quality=88)
                if is_fake:
                    got_fake += 1
                else:
                    got_real += 1
                if (got_real + got_fake) % 100 == 0:
                    print(f"  {split}: +{got_real}R +{got_fake}A", flush=True)
            except Exception as e:
                print("  skip:", str(e)[:100])
        if split == "train":
            state["train_saved"] += got_real + got_fake
        else:
            state["test_saved"] += got_real + got_fake
        state_p.write_text(json.dumps(state))
        print(f"{split}: saved +{got_real}R +{got_fake}A (total {state})")
    print("Done:", json.dumps(state))


if __name__ == "__main__":
    main()
