# 4.1.7.7 服务保活接口

import logging

from fastapi import APIRouter

from app.schemas import KeepAliveRequest, KeepAliveResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analysis/api/v1/keepAlive", response_model=KeepAliveResponse)
async def keep_alive(req: KeepAliveRequest):
    """服务保活：接收保活心跳"""
    return KeepAliveResponse(
        resultCode="200",
        resultValue={"devId": f"{req.devIP}:{req.devPort}"},
        resultHint="keep alive",
    )
