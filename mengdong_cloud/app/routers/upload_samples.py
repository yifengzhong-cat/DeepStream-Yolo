# 4.1.7.6 样本数据回传接口

import logging

from fastapi import APIRouter

from app.schemas import UploadSamplesRequest, UploadSamplesResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/service/uploadSamples", response_model=UploadSamplesResponse)
async def upload_samples(req: UploadSamplesRequest):
    """样本数据回传：接收样本文件信息"""
    logger.info("收到样本上传请求: fileId=%s, md5=%s", req.fileId, req.md5)

    return UploadSamplesResponse(
        resultCode="200",
        resultValue={"fileId": req.fileId},
        resultHint="接收样本成功",
    )
