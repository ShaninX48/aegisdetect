"""Forensic-style PDF report - PRD section 20."""
import io
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def build_report_pdf(rec: dict) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 20 * mm

    def line(text, size=11, bold=False, gap=6 * mm, color=(0, 0, 0)):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColorRGB(*color)
        c.drawString(20 * mm, y, text[:100])
        y -= gap

    # Header
    line("AI MEDIA ANALYSIS REPORT", 16, bold=True, gap=8 * mm)
    line(f"Analysis ID: {rec['analysis_id']}", 11, gap=5 * mm)
    line(f"Date (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", 10, gap=8 * mm)

    # Classification block
    label = {"likely_ai": "LIKELY AI-GENERATED", "likely_authentic": "LIKELY AUTHENTIC",
             "inconclusive": "INCONCLUSIVE"}.get(rec["result"], rec["result"].upper())
    color = (0.7, 0, 0) if rec["result"] == "likely_ai" else (0, 0.45, 0) if rec["result"] == "likely_authentic" else (0.6, 0.5, 0)
    line(f"Classification: {label}", 14, bold=True, color=color)
    line(f"AI probability: {rec['ai_probability']:.2%}  |  Authentic: {rec['authentic_probability']:.2%}", 11)

    y -= 2 * mm
    line("Media Information", 12, bold=True)
    line(f"Media type: {rec['media_type']}  |  Extension: .{rec.get('file_ext', '?')}  |  Size: {rec.get('file_size_bytes', 0)} bytes", 10)
    # hash may exceed 100 chars -> split
    line(f"SHA-256: {rec.get('file_hash_sha256', '')[:64]}", 9)
    line(f"         {rec.get('file_hash_sha256', '')[64:]}", 9)
    meta = rec.get("metadata", {}) or {}
    if rec.get("media_type") == "audio":
        line(f"Duration: {meta.get('duration_sec', '?')}s  |  Sample rate: {meta.get('sample_rate', '?')} Hz  |  Channels: {meta.get('channels', '?')}", 10)
    else:
        line(f"Dimensions: {meta.get('width', '?')}x{meta.get('height', '?')}  |  Format: {meta.get('format', '?')}", 10)
    exif = meta.get("exif") or {}
    line(f"EXIF entries: {len(exif)}" + (f"  (e.g. {', '.join(list(exif)[:4])})" if exif else "  (stripped/missing - common in AI images)"), 10)

    y -= 2 * mm
    line("Detection Indicators", 12, bold=True)
    for ind in rec.get("indicators", [])[:8]:
        line(f"- {ind.get('label', ind.get('code'))} ({float(ind.get('score', 0)):.0%})", 10, gap=5 * mm)

    y -= 2 * mm
    line("Model & Limitations", 12, bold=True)
    line(f"Model: {rec.get('model_version', '?')}  |  Processing: {rec.get('processing_time_ms', '?')} ms", 10)
    line("This result is probabilistic and must not be treated as", 10, gap=5 * mm)
    line("definitive proof of authenticity or fraud.", 10, bold=True, gap=5 * mm)
    line("Recommended: combine with metadata, provenance (C2PA),", 9)
    line("and human expert review for high-stakes decisions.", 9)

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(20 * mm, 15 * mm, f"AegisDetect forensic-style screening report - {rec['analysis_id']}")
    c.showPage()
    c.save()
    return buf.getvalue()
