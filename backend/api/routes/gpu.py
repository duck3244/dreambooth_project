from fastapi import APIRouter

from .. import gpu as gpu_info
from ..schemas import GPUInfo

router = APIRouter(tags=["gpu"])


@router.get("/gpu", response_model=GPUInfo)
async def get_gpu() -> GPUInfo:
    return gpu_info.get_info()
