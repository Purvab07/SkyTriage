"""Classification endpoint: defect text in, aircraft system + calibrated confidence out."""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.triage import ClassifyRequest, ClassifyResponse
from app.services.classify import ClassifierService, ModelNotAvailableError, get_classifier

router = APIRouter(prefix="/v1", tags=["triage"])


def classifier_dependency() -> ClassifierService:
    try:
        return get_classifier()
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Predict which aircraft system a defect belongs to",
)
def classify(
    request: ClassifyRequest,
    classifier: Annotated[ClassifierService, Depends(classifier_dependency)],
) -> ClassifyResponse:
    # A plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
    # CPU work of the model never blocks other requests.
    return classifier.classify(request.text)