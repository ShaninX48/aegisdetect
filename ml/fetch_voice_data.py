"""Fetch web voice data: LibriSpeech (real) + Edge-TTS neural voices (AI).
Usage: python fetch_voice_data.py --real-n 25 --ai-n 12 --out ./voice_data
Requires: pip install edge-tts  (LibriSpeech streams from HF, no download of full 30GB)
"""
import argparse
import asyncio
import wave
from pathlib import Path

import numpy as np


def save_wav(path: Path, x: np.ndarray, sr: int = 16000):
    x = np.clip(x, -1, 1)
    pcm = (x * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm)


def fetch_real(out: Path, n: int):
    """LibriSpeech test-clean direct mirror (openslr.org) + soundfile decode.
    No torchcodec/ffmpeg needed. One-time ~312MB tar, cached in _librispeech/."""
    import tarfile
    import urllib.request
    import soundfile as sf
    cache = Path("_librispeech")
    tar = cache / "test-clean.tar.gz"
    clip_dir = cache / "LibriSpeech" / "test-clean"
    (out / "real").mkdir(parents=True, exist_ok=True)
    if not clip_dir.exists():
        cache.mkdir(parents=True, exist_ok=True)
        if not tar.exists():
            url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
            print(f"Downloading LibriSpeech test-clean (~312MB) from {url} ...")
            urllib.request.urlretrieve(url, tar)
            print("Download done.")
        print("Extracting...")
        with tarfile.open(tar, "r:gz") as t:
            t.extractall(cache)
    flacs = sorted(clip_dir.rglob("*.flac"))
    print(f"Found {len(flacs)} clips, taking {n}")
    got = 0
    for p in flacs:
        if got >= n:
            break
        try:
            x, sr = sf.read(str(p), dtype="float64", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000 and len(x) > 0:
                idx = np.clip((np.arange(int(len(x) * 16000 / sr)) * sr / 16000).astype(int), 0, len(x) - 1)
                x = x[idx]
            peak = np.abs(x).max()
            x = x / peak if peak > 0 else x
            if len(x) < 16000:
                continue
            save_wav(out / "real" / f"librispeech_{got:03d}.wav", x[:10 * 16000])
            got += 1
            print(f"real {got}/{n}", flush=True)
        except Exception as e:
            print("skip:", str(e)[:100])
    print(f"Real saved: {got} in {out/'real'}")
    return got


AI_TEXTS = [
    ("en-US-GuyNeural", "Please verify your account immediately to avoid suspension."),
    ("en-US-JennyNeural", "Your package has arrived at the collection point."),
    ("en-GB-SoniaNeural", "The meeting has been rescheduled to Monday morning."),
    ("en-AU-NatashaNeural", "Remember to bring your identification documents."),
    ("bn-BD-NabanitaNeural", "আমার ভোটার আইডি কার্ড হারিয়ে গেছে, সাহায্য করুন।"),
    ("bn-BD-PradeepNeural", "আগামীকাল সকাল দশটায় অফিসে আসুন।"),
    ("en-IN-NeerjaNeural", "Your one-time password is four five six seven."),
    ("en-US-DavisNeural", "This is a security alert regarding your recent login."),
    ("en-GB-RyanNeural", "Kindly confirm your attendance by Friday evening."),
    ("en-US-AriaNeural", "The weather today will be sunny with light winds."),
    ("en-IE-EmilyNeural", "She sells seashells by the seashore every day."),
    ("en-NZ-MollyNeural", "The quick brown fox jumps over the lazy dog."),
]


async def _edge_one(voice: str, text: str, path: Path):
    import edge_tts
    tts = edge_tts.Communicate(text, voice)
    data = b""
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    # edge returns mp3; decode via... keep mp3? Backend needs WAV.
    # Use simple mp3->wav via imageio? Instead request raw: edge-tts only mp3.
    # Save mp3 then note conversion need. Better: use `edge-tts --write-file` CLI which also mp3.
    # Workaround: save mp3 bytes; caller converts with pydub if present else keep mp3.
    return data


def fetch_ai(out: Path, n: int):
    try:
        import edge_tts  # noqa
    except ImportError:
        raise SystemExit("pip install edge-tts  (then re-run)")
    import subprocess
    (out / "ai").mkdir(parents=True, exist_ok=True)
    got = 0
    for i in range(min(n, len(AI_TEXTS))):
        voice, text = AI_TEXTS[i]
        mp3 = out / "ai" / f"edge_{got:03d}.mp3"
        wav = out / "ai" / f"edge_{got:03d}.wav"
        # CLI writes mp3 (newer versions: --write-media, older: --write-file)
        r = subprocess.run(["python", "-m", "edge_tts", "--voice", voice,
                            "--text", text, "--write-media", str(mp3)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not mp3.exists():
            r = subprocess.run(["python", "-m", "edge_tts", "--voice", voice,
                                "--text", text, "--write-file", str(mp3)],
                               capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not mp3.exists():
            print("edge-tts fail:", voice, r.stderr[:150])
            continue
        # mp3 -> wav 16k mono: try pydub/ffmpeg, else imageio-ffmpeg, else keep mp3
        converted = False
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(str(mp3)).set_frame_rate(16000).set_channels(1)
            seg.export(str(wav), format="wav")
            converted = True
        except Exception:
            try:
                import imageio_ffmpeg, subprocess as sp
                exe = imageio_ffmpeg.get_ffmpeg_exe()
                sp.run([exe, "-y", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)],
                       capture_output=True, timeout=120, check=True)
                converted = True
            except Exception as e:
                print("mp3->wav needs ffmpeg; kept mp3:", str(e)[:100])
        if converted:
            mp3.unlink(missing_ok=True)
        got += 1
        print(f"ai {got}/{n} ({voice})", flush=True)
    print(f"AI saved: {got} in {out/'ai'}")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-n", type=int, default=25)
    ap.add_argument("--ai-n", type=int, default=12)
    ap.add_argument("--out", default="./voice_data")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-ai", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.skip_real:
        fetch_real(out, args.real_n)
    if not args.skip_ai:
        fetch_ai(out, args.ai_n)
    print("Done. Next: python voice_baseline.py --data-dir ./voice_data --out ./voice_real_out")


if __name__ == "__main__":
    main()
