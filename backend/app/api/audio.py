import time
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.audio_service import (
    validate_audio, sha256_hex, extract_audio_metadata,
    classify_audio_dummy, DISCLAIMER, MAX_SECONDS,
)
from app.services.store import DB, next_id

try:
    from app.services.voice_model import predict_wav, MODEL_VERSION as VOICE_VER
    _HAS_VOICE = True
except Exception:
    predict_wav = None
    _HAS_VOICE = False

router = APIRouter()
MODEL_VERSION = VOICE_VER if _HAS_VOICE else "AegisVoice stub-0.1.0"


@router.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        content = await file.read()
        ext = validate_audio(file.filename or "upload", content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    file_hash = sha256_hex(content)
    meta = extract_audio_metadata(content, ext)
    dur = meta.get("duration_sec")
    if isinstance(dur, (int, float)) and dur > MAX_SECONDS:
        raise HTTPException(status_code=400, detail=f"Audio too long ({dur}s). Max {MAX_SECONDS}s for MVP.")

    result, ai_prob, indicators = classify_audio_dummy(file_hash)
    used_version = MODEL_VERSION
    if _HAS_VOICE and ext == "wav":
        try:
            result, ai_prob, indicators = predict_wav(content)
            used_version = VOICE_VER
        except Exception:
            used_version = MODEL_VERSION
    elif ext != "wav":
        indicators = [{"code": "wav_only", "label": "Voice baseline needs WAV for now - convert MP3/M4A to WAV", "score": ai_prob}]

    analysis_id = next_id()
    record = {
        "analysis_id": analysis_id,
        "media_type": "audio",
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
