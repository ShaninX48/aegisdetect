from fastapi import APIRouter, HTTPException, Response
from app.services.store import DB

router = APIRouter()


@router.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str):
    rec = DB.get(analysis_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return rec


@router.get("/analysis/{analysis_id}/report")
def get_report(analysis_id: str, format: str = "pdf"):
    """PDF forensic report (PRD-20). ?format=text for plain-text fallback."""
    rec = DB.get(analysis_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if format == "text":
        text = (
            "AI MEDIA ANALYSIS REPORT\n"
            f"Analysis ID: {rec['analysis_id']}\n"
            f"Media Type: {rec['media_type']}\n"
            f"SHA-256: {rec['file_hash_sha256']}\n"
            f"Classification: {rec['result'].upper()}\n"
            f"AI probability: {rec['ai_probability']}\n"
            f"Model: {rec['model_version']}\n"
            f"Important: {rec['disclaimer']}\n"
        )
        return Response(content=text, media_type="text/plain")
    from app.services.report import build_report_pdf
    pdf = build_report_pdf(rec)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={analysis_id}.pdf"},
    )
