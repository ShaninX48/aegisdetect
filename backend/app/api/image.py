import time
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.image_service import (
    validate_image, sha256_hex, extract_metadata, DISCLAIMER,
)
from app.services.store import DB, next_id

try:
    from app.services.v4_vision import predict as v4_predict, MODEL_VERSION as V4_VER
    _HAS_V4 = True
except Exception:
    v4_predict = None
    _HAS_V4 = False

try:
    from app.services.cnn_vision import predict as cnn_predict, MODEL_VERSION as CNN_VER
    _HAS_CNN = True
except Exception:
    cnn_predict = None
    _HAS_CNN = False

try:
    from app.services.deep_vision import predict as deep_predict, MODEL_VERSION as DEEP_VER
    _HAS_DEEP = True
except Exception:
    deep_predict = None
    _HAS_DEEP = False

try:
    from app.services.aegis_vision import predict as vision_predict
    _HAS_MODEL = True
except Exception:
    vision_predict = None
    _HAS_MODEL = False

from app.services.image_service import classify_dummy

router = APIRouter()
if _HAS_V4:
    MODEL_VERSION = V4_VER
elif _HAS_CNN:
    MODEL_VERSION = CNN_VER
elif _HAS_DEEP:
    MODEL_VERSION = DEEP_VER
elif _HAS_MODEL:
    MODEL_VERSION = "AegisVision v1.0-baseline"
else:
    MODEL_VERSION = "AegisVision stub-0.1.0"


@router.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        content = await file.read()
        ext = validate_image(file.filename or "upload", content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    file_hash = sha256_hex(content)
    meta = extract_metadata(content)
    used_version = MODEL_VERSION
    if _HAS_V4:
        try:
            result, ai_prob, indicators = v4_predict(content)
            used_version = V4_VER
        except Exception:
            result, ai_prob, indicators = classify_dummy(file_hash)
            used_version = "AegisVision stub-0.1.0"
    elif _HAS_CNN:
        try:
            result, ai_prob, indicators = cnn_predict(content)
            used_version = CNN_VER
        except Exception:
            result, ai_prob, indicators = classify_dummy(file_hash)
            used_version = "AegisVision stub-0.1.0"
    elif _HAS_DEEP:
        try:
            result, ai_prob, indicators = deep_predict(content)
        except Exception:
            result, ai_prob, indicators = classify_dummy(file_hash)
            used_version = "AegisVision stub-0.1.0"
    elif _HAS_MODEL:
        try:
            result, ai_prob, indicators = vision_predict(content)
            used_version = "AegisVision v1.0-baseline"
        except Exception:
            result, ai_prob, indicators = classify_dummy(file_hash)
            used_version = "AegisVision stub-0.1.0"
    else:
        result, ai_prob, indicators = classify_dummy(file_hash)
        used_version = "AegisVision stub-0.1.0"

    analysis_id = next_id()
    record = {
        "analysis_id": analysis_id,
        "media_type": "image",
        "result": result,
        "ai_probability": ai_prob,
        "authentic_probability": round(1.0 - ai_prob, 2),
        "indicators": indicators,
        "model_version": used_version,
        "file_hash_sha256": file_hash,
        "file_size_bytes": len(content),
        "file_ext": ext,
        "metadata": meta,
        "processing_time_ms": int((time.time() - t0) * 1000),
        "disclaimer": DISCLAIMER,
    }
    DB[analysis_id] = record
    return record
