"""FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import uvloop  # type: ignore[import-not-found]

    uvloop.install()
except ImportError:  # pragma: no cover
    pass

from backend.api import rest, ws
from backend.config import settings
from backend.log import get_logger, setup_logging
from backend.runtime import get_runtime, shutdown_runtime

log = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("INFO", settings.logs_dir)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    log.info("opus starting; mode=%s host=%s port=%d", settings.mode.value, settings.host, settings.port)
    runtime = get_runtime()
    await runtime.start()
    try:
        yield
    finally:
        log.info("opus shutting down")
        await ws.shutdown()
        await shutdown_runtime()


app = FastAPI(title="opus", version="0.1.0", lifespan=lifespan)
app.include_router(rest.router)
app.include_router(ws.router)


if FRONTEND_DIR.exists():

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
else:

    @app.get("/", include_in_schema=False)
    async def index() -> JSONResponse:  # pragma: no cover
        return JSONResponse({"message": "frontend not found", "api": "/api/status"})


def run() -> None:
    """Used by ``python -m backend.main`` for local runs."""
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=bool(int(os.environ.get("OPUS_RELOAD", "0"))),
    )


if __name__ == "__main__":
    run()
