from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "models_loaded": False, "model_version": "stub-0.1.0"}
