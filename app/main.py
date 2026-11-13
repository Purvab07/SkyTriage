from fastapi import FastAPI

from app.api import routes_health
from app.config import settings

app = FastAPI(
    title="SkyTriage",
    version="0.1.0",
    description="Advisory safety and reliability flags for small aviation operators.",
)

app.include_router(routes_health.router)


@app.get("/")
async def root() -> dict:
    return {"name": "SkyTriage", "env": settings.app_env, "docs": "/docs"}
