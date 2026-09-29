"""Backend plug-in for mixed-trained TinyCNN (fast, real-world tuned).
Copy mixed_cnn.pt next to this file. Lazy torch load; graceful fallback.
"""
import io

_NET = None
from pathlib import Path

_MODEL_PATH = Path(__file__).parent / "mixed_cnn.pt"
MODEL_VERSION = "AegisVision v3.0-mixed-cnn"
SIZE = 32


def _arch():
    import torch.nn as nn

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

    return TinyCNN()


def _load():
    global _NET
    if _NET is None:
        import torch
        _NET = _arch()
        _NET.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu"))
        _NET.eval()
    return _NET


def predict(content: bytes):
    import numpy as np
    import torch
    from PIL import Image
    try:
        net = _load()
    except Exception as e:
        return "inconclusive", 0.5, [
            {"code": "cnn_unavailable", "label": f"Mixed CNN not loaded ({e})", "score": 0.5}
        ]
    img = Image.open(io.BytesIO(content)).convert("RGB").resize((SIZE, SIZE))
    arr = np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0
    with torch.no_grad():
        ai_prob = round(float(torch.sigmoid(net(torch.tensor(arr).unsqueeze(0))).item()), 4)
    result = "likely_ai" if ai_prob > 0.70 else ("likely_authentic" if ai_prob < 0.40 else "inconclusive")
    return result, ai_prob, [
        {"code": "mixed_cnn", "label": "Mixed CNN (SynthCIFAR + real-world aug, 32x32)", "score": ai_prob}
    ]
