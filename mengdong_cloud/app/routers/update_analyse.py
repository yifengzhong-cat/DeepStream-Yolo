# 4.1.7.8 更新分析ID接口

import logging

from fastapi import APIRouter

from app.schemas import UpdateAnalyseIDRequest, UpdateAnalyseIDResponse
from app.tasks.video_processor import video_task_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/analysis/api/v1/updateAnalyseID", response_model=UpdateAnalyseIDResponse
)
async def update_analyse_id(req: UpdateAnalyseIDRequest):
    """更新分析ID：重新组装videoUrl并拉流"""
    success = video_task_manager.update_analyse_id(req.oldAnalyseId, req.newAnalyseId)

    if success:
        return UpdateAnalyseIDResponse(
            resultCode="200",
            resultValue={"newAnalyseId": req.newAnalyseId},
            resultHint="更新成功",
        )

    return UpdateAnalyseIDResponse(
        resultCode="404",
        resultValue=None,
        resultHint="未找到原分析ID对应的任务",
    )
