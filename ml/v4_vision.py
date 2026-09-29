"""Backend plug-in for v4 (frozen ResNet18 + linear probe, 224px, structure-level).
Install: copy this file -> backend/app/services/v4_vision.py + v4_probe.pt beside it.
Requires: torchvision (CPU). First call loads ResNet (~45MB cached) + probe.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_NET = _PROBE = None
_HERE = Path(__file__).parent
_PROBE_PATH = _HERE / "v4_probe.pt"
MODEL_VERSION = "AegisVision v4.0-resnet-probe"
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def _load():
    global _NET, _PROBE
    if _NET is None:
        from torchvision import models
        try:
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            net = models.resnet18(weights=None)
        net.fc = nn.Identity()
        for p in net.parameters():
            p.requires_grad_(False)
        net.eval()
        ckpt = torch.load(_PROBE_PATH, map_location="cpu")
        raw = ckpt["probe"]
        # train saves Probe.fc.* keys -> map to plain Linear
        state = {k.split(".", 1)[1] if "." in k else k: v for k, v in raw.items()}
        probe = nn.Linear(512, 1)
        probe.load_state_dict(state)
        probe.eval()
        _NET, _PROBE = net, probe
    return _NET, _PROBE


def _prep(content: bytes) -> torch.Tensor:
    from PIL import Image
    img = Image.open(__import__("io").BytesIO(content)).convert("RGB")
    s = max(1, round(256 / min(img.size)))
    img = img.resize((img.width * s, img.height * s), Image.BILINEAR)
    w, h = img.size
    img = img.crop(((w - SIZE) // 2, (h - SIZE) // 2, (w + SIZE) // 2, (h + SIZE) // 2))
    arr = np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch.tensor((arr - MEAN) / STD).unsqueeze(0).float()


def predict(content: bytes):
    try:
        net, probe = _load()
    except Exception as e:
        return "inconclusive", 0.5, [
            {"code": "v4_unavailable", "label": f"v4 not loaded ({str(e)[:100]})", "score": 0.5}
        ]
    with torch.no_grad():
        ai_prob = round(float(torch.sigmoid(probe(net(_prep(content)))).item()), 4)
    result = "likely_ai" if ai_prob > 0.70 else ("likely_authentic" if ai_prob < 0.40 else "inconclusive")
    return result, ai_prob, [
        {"code": "resnet_probe", "label": "ResNet18 structure probe (224px, ImageNet)", "score": ai_prob}
    ]
