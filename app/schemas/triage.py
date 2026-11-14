"""Request and response shapes for classification.

These Pydantic models are the API contract: FastAPI uses them to validate input,
reject bad requests with a clear error, and generate the /docs page automatically.
"""
from pydantic import BaseModel, Field, field_validator

ADVISORY = (
    "Advisory only. Confirm with the official AD, approved maintenance data and a "
    "certified mechanic before acting."
)


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The defect write-up, as a mechanic would type it.",
        examples=["MLG ACTUATOR LEAKING HYD FLUID. REPLACED SEAL."],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = " ".join(value.split())  # collapse spaces and line breaks
        if len(value) < 3:
            raise ValueError("text must contain at least 3 non-space characters")
        return value


class ChapterOption(BaseModel):
    chapter: str = Field(description="ATA chapter code, e.g. '32'")
    name: str = Field(description="Readable system name, e.g. 'Landing Gear'")
    probability: float = Field(ge=0, le=1, description="Calibrated probability")


class ClassifyResponse(BaseModel):
    top: ChapterOption = Field(description="The most likely aircraft system")
    prediction_set: list[ChapterOption] = Field(
        description="Every system that could be the answer, best first. "
        "Contains the true answer about `coverage_target` of the time."
    )
    coverage_target: float = Field(description="Guaranteed hit rate of prediction_set")
    confident: bool = Field(
        description="True when the set has one answer and the probability is above "
        "the cascade threshold, i.e. safe for the fast path."
    )
    model_version: str
    advisory: str = ADVISORY