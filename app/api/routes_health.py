from fastapi import APIRouter

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health")
async def health() -> dict:
    # Later: also report model, index and prompt versions in use.
    return {"status": "ok"}
