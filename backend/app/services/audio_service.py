"""Audio pipeline services - Phase 2 skeleton.
Real model TODO (Week 4): mel-spectrogram CNN or Wav2Vec2+AASIST trained on ASVspoof2019 LA.
"""
import hashlib
import io
import wave

ALLOWED_EXTS = {"wav", "mp3", "m4a"}
MAX_BYTES = 20 * 1024 * 1024  # 20MB
MAX_SECONDS = 60
DISCLAIMER = "Probabilistic estimate, not definitive proof."


def validate_audio(filename: str, content: bytes) -> str:
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported format .{ext}. Allowed: WAV/MP3/M4A")
    if len(content) > MAX_BYTES:
        raise ValueError("File too large. Max 20MB for MVP.")
    if ext == "wav":
        try:
            with wave.open(io.BytesIO(content), "rb") as w:
                if w.getnframes() <= 0:
                    raise ValueError("Empty WAV file.")
        except wave.Error:
            raise ValueError("Invalid or corrupted WAV file.")
    return ext


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extract_audio_metadata(content: bytes, ext: str) -> dict:
    """Duration / sample-rate without executing uploads. WAV native, MP3/M4A via mutagen if present."""
    meta: dict = {"format": ext.upper()}
    try:
        if ext == "wav":
            with wave.open(io.BytesIO(content), "rb") as w:
                frames, rate = w.getnframes(), w.getframerate()
                meta.update({
                    "sample_rate": rate,
                    "channels": w.getnchannels(),
                    "duration_sec": round(frames / max(1, rate), 2),
                })
            return meta
        try:
            import mutagen

            f = mutagen.File(io.BytesIO(content))
            if f is not None and hasattr(f, "info"):
                meta["duration_sec"] = round(float(getattr(f.info, "length", 0) or 0), 2)
                if hasattr(f.info, "sample_rate"):
                    meta["sample_rate"] = int(f.info.sample_rate)
                if hasattr(f.info, "channels"):
                    meta["channels"] = int(f.info.channels)
            else:
                meta["note"] = "duration unknown (install mutagen for MP3/M4A tags)"
        except ImportError:
            meta["note"] = "duration unknown (install mutagen for MP3/M4A tags)"
    except Exception as e:
        meta["error"] = str(e)[:200]
    return meta


def classify_audio_dummy(file_hash: str) -> tuple[str, float, list]:
    """Placeholder: always inconclusive zone. Replace with AegisVoice v1.0 in Week 4."""
    pseudo = (int(file_hash[:8], 16) % 100) / 100.0
    ai_prob = round(0.35 + pseudo * 0.3, 2)  # 0.35-0.65
    result = "likely_ai" if ai_prob > 0.70 else ("likely_authentic" if ai_prob < 0.40 else "inconclusive")
    return result, ai_prob, [
        {"code": "stub_voice", "label": "Placeholder voice model - real detection not yet loaded", "score": ai_prob}
    ]
