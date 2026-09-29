import hashlib
import io
from fastapi import UploadFile
from PIL import Image, ExifTags

ALLOWED_EXTS = {"jpg", "jpeg", "png", "webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10MB

DISCLAIMER = "Probabilistic estimate, not definitive proof."


def validate_image(filename: str, content: bytes) -> str:
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported format .{ext}. Allowed: JPG/PNG/WEBP")
    if len(content) > MAX_BYTES:
        raise ValueError("File too large. Max 10MB for MVP.")
    # Verify it is actually an image (never execute uploads)
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
    except Exception:
        raise ValueError("Invalid or corrupted image file.")
    return ext


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extract_metadata(content: bytes) -> dict:
    """Basic EXIF + size info. No file is executed."""
    try:
        img = Image.open(io.BytesIO(content))
        w, h = img.size
        exif = {}
        raw = img.getexif() if hasattr(img, "getexif") else None
        if raw:
            for k, v in raw.items():
                tag = ExifTags.TAGS.get(k, str(k))
                exif[str(tag)] = str(v)[:200]
        return {"width": w, "height": h, "mode": img.mode, "format": img.format, "exif": exif}
    except Exception as e:
        return {"error": str(e)}


def classify_dummy(file_hash: str) -> tuple[str, float, list]:
    """Deterministic placeholder so UI can be built before real model.
    Replace with AegisVision v1.0 in Week 2.
    """
    # Use hash to give stable pseudo-probability (NOT a real detection)
    pseudo = (int(file_hash[:8], 16) % 100) / 100.0
    ai_prob = round(0.3 + pseudo * 0.4, 2)  # always 0.30-0.70 -> inconclusive zone
    if ai_prob > 0.70:
        result = "likely_ai"
    elif ai_prob < 0.40:
        result = "likely_authentic"
    else:
        result = "inconclusive"
    indicators = [
        {"code": "stub_model", "label": "Placeholder model — real detection not yet loaded", "score": ai_prob}
    ]
    return result, ai_prob, indicators
