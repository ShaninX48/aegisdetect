"""Backend plug-in for deep model. Copy to backend/app/services/deep_vision.py.
Lazy-loads HF pipeline; falls back gracefully if torch/transformers missing.
"""
import io

_PIPE = None
MODEL_ID = "umm-maybe/AI-image-detector"
MODEL_VERSION = "AegisVision v2.0-ViT-deep"


def _load():
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _PIPE = pipeline("image-classification", MODEL_ID, device=device)
    return _PIPE


def _ai_prob(outputs) -> float:
    ai, human = 0.0, 0.0
    for o in outputs:
        lab, s = o["label"].lower(), float(o["score"])
        if any(k in lab for k in ("artific", "ai", "fake", "generat", "synth")):
            ai += s
        elif any(k in lab for k in ("human", "real", "authentic", "natural")):
            human += s
    if ai + human == 0:
        return float(outputs[0]["score"])
    return ai / (ai + human)


def predict(content: bytes):
    from PIL import Image
    try:
        pipe = _load()
    except Exception as e:
        return "inconclusive", 0.5, [
            {"code": "deep_unavailable", "label": f"Deep model not loaded ({e})", "score": 0.5}
        ]
    img = Image.open(io.BytesIO(content)).convert("RGB")
    ai_prob = round(_ai_prob(pipe(img)), 4)
    result = "likely_ai" if ai_prob > 0.70 else ("likely_authentic" if ai_prob < 0.40 else "inconclusive")
    return result, ai_prob, [
        {"code": "vit_deep", "label": "ViT deep detector (umm-maybe, calibrated thresholds)", "score": ai_prob}
    ]
