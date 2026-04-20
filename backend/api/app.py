"""FastAPI application.

Run with:
    PYTHONPATH=. uvicorn backend.api.app:app --host 127.0.0.1 --port 8000 --reload

The API process deliberately avoids importing torch. All training and
inference is fanned out to subprocesses (backend.api.workers.*) so the
HTTP layer stays responsive and GPU state is owned by exactly one
process at a time.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import gpu, settings
from .inference_manager import InferenceManager
from .job_manager import JobManager
from .routes import datasets as datasets_routes
from .routes import gpu as gpu_routes
from .routes import health as health_routes
from .routes import inference as inference_routes
from .routes import models as models_routes
from .routes import train as train_routes


logger = logging.getLogger("dreambooth.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    gpu.init()

    manager = JobManager()
    manager.reconcile_on_startup()
    app.state.job_manager = manager

    inference_manager = InferenceManager()
    inference_manager.reconcile_on_startup()
    app.state.inference_manager = inference_manager

    # Background reap task — keeps state.json current even if no HTTP
    # request happens to land during/after a job's exit.
    stop_event = asyncio.Event()

    async def reaper():
        while not stop_event.is_set():
            try:
                manager.reap_finished()
                inference_manager.reap_finished()
            except Exception as e:
                logger.warning("reap failed: %s", e)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

    reaper_task = asyncio.create_task(reaper())

    try:
        yield
    finally:
        stop_event.set()
        reaper_task.cancel()
        try:
            await reaper_task
        except (asyncio.CancelledError, Exception):
            pass
        gpu.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="DreamBooth API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_routes.router, prefix="/api")
    app.include_router(gpu_routes.router, prefix="/api")
    app.include_router(datasets_routes.router, prefix="/api")
    app.include_router(train_routes.router, prefix="/api")
    app.include_router(models_routes.router, prefix="/api")
    app.include_router(inference_routes.router, prefix="/api")

    return app


app = create_app()
