from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import routes_health, routes_triage
from app.config import settings

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="SkyTriage",
    version="0.1.0",
    description="Advisory safety and reliability flags for small aviation operators.",
)

app.include_router(routes_health.router)
app.include_router(routes_triage.router)


@app.get("/")
async def root() -> dict:
    return {"name": "SkyTriage", "env": settings.app_env, "docs": "/docs", "demo": "/demo"}


@app.get("/demo", include_in_schema=False)
async def demo() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")