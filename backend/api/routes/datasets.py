"""Dataset management: create, list, upload images, delete."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from .. import settings
from ..paths import UnsafePathError, resolve_under, safe_component, safe_filename
from ..schemas import (
    DatasetCreateResponse,
    DatasetInfo,
    DatasetList,
    DatasetUploadResponse,
    OkResponse,
)
from ..state import read_json, write_json_atomic

router = APIRouter(prefix="/datasets", tags=["datasets"])

META_FILENAME = "meta.json"


def _dataset_meta_path(dataset_id: str) -> Path:
    return resolve_under(settings.DATASETS_DIR, dataset_id, META_FILENAME)


def _dataset_images_dir(dataset_id: str) -> Path:
    return resolve_under(settings.DATASETS_DIR, dataset_id, "images")


def _list_image_files(ds_id: str) -> List[Path]:
    images_dir = _dataset_images_dir(ds_id)
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in settings.ALLOWED_IMAGE_SUFFIXES
    )


def _to_info(ds_id: str, meta: dict) -> DatasetInfo:
    files = _list_image_files(ds_id)
    total = sum(p.stat().st_size for p in files)
    return DatasetInfo(
        id=ds_id,
        name=meta.get("name", ds_id),
        image_count=len(files),
        total_bytes=total,
        created_at=meta.get("created_at", 0),
    )


@router.get("", response_model=DatasetList)
async def list_datasets() -> DatasetList:
    settings.ensure_dirs()
    out: List[DatasetInfo] = []
    for d in sorted(settings.DATASETS_DIR.iterdir()) if settings.DATASETS_DIR.exists() else []:
        if not d.is_dir():
            continue
        meta = read_json(d / META_FILENAME)
        if meta:
            try:
                out.append(_to_info(d.name, meta))
            except UnsafePathError:
                continue
    out.sort(key=lambda x: x.created_at, reverse=True)
    return DatasetList(datasets=out)


@router.post("", response_model=DatasetCreateResponse)
async def create_dataset(name: str = Form(...)) -> DatasetCreateResponse:
    settings.ensure_dirs()
    # Name is display-only; id is server-generated.
    name = name.strip()
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name is required")
    if len(name) > 128:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name too long (max 128)")

    ds_id = uuid.uuid4().hex[:12]
    ds_dir = resolve_under(settings.DATASETS_DIR, ds_id)
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "images").mkdir(exist_ok=True)

    meta = {"id": ds_id, "name": name, "created_at": time.time()}
    write_json_atomic(ds_dir / META_FILENAME, meta)
    return DatasetCreateResponse(id=ds_id, name=name)


@router.post("/{dataset_id}/images", response_model=DatasetUploadResponse)
async def upload_images(
    dataset_id: str,
    files: List[UploadFile] = File(...),
) -> DatasetUploadResponse:
    try:
        ds_id = safe_component(dataset_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    meta = read_json(resolve_under(settings.DATASETS_DIR, ds_id, META_FILENAME))
    if not meta:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")

    images_dir = _dataset_images_dir(ds_id)
    images_dir.mkdir(parents=True, exist_ok=True)

    added: List[str] = []
    skipped: List[str] = []
    for upload in files:
        try:
            fname = safe_filename(upload.filename or "")
        except UnsafePathError:
            skipped.append(upload.filename or "<unnamed>")
            continue

        if Path(fname).suffix.lower() not in settings.ALLOWED_IMAGE_SUFFIXES:
            skipped.append(fname)
            continue

        dest = resolve_under(settings.DATASETS_DIR, ds_id, "images", fname)
        size = 0
        try:
            with open(dest, "wb") as f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > settings.MAX_UPLOAD_BYTES:
                        f.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file too large: {fname} (max {settings.MAX_UPLOAD_BYTES} bytes)",
                        )
                    f.write(chunk)
            added.append(fname)
        except HTTPException:
            raise
        except Exception:
            dest.unlink(missing_ok=True)
            skipped.append(fname)

    return DatasetUploadResponse(
        id=ds_id,
        added=added,
        skipped=skipped,
        image_count=len(_list_image_files(ds_id)),
    )


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str) -> DatasetInfo:
    try:
        ds_id = safe_component(dataset_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    meta = read_json(_dataset_meta_path(ds_id))
    if not meta:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")
    return _to_info(ds_id, meta)


@router.delete("/{dataset_id}", response_model=OkResponse)
async def delete_dataset(dataset_id: str) -> OkResponse:
    try:
        ds_id = safe_component(dataset_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    ds_dir = resolve_under(settings.DATASETS_DIR, ds_id)
    if not ds_dir.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dataset not found")
    shutil.rmtree(ds_dir)
    return OkResponse()


def dataset_images_path(dataset_id: str) -> Path:
    """Helper for the train route to resolve a validated dataset's images path."""
    return _dataset_images_dir(safe_component(dataset_id))
