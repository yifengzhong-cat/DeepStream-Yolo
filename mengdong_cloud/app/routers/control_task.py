# 4.1.7.3 分析任务控制接口

import logging

from fastapi import APIRouter

from app.schemas import ControlTaskRequest, ControlTaskResponse
from app.tasks.video_processor import video_task_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/service/controlTask", response_model=ControlTaskResponse)
async def control_task(req: ControlTaskRequest):
    """分析任务控制：启停和删除任务"""
    if req.command == 0:
        # 停止
        success = video_task_manager.stop_task(req.analyseId)
        if success:
            return ControlTaskResponse(
                resultCode="200",
                resultValue=None,
                resultHint="任务已停止",
            )
        return ControlTaskResponse(
            resultCode="404",
            resultValue=None,
            resultHint="未找到指定任务",
        )

    elif req.command == 1:
        # 开始（恢复）
        task = video_task_manager.get_task(req.analyseId)
        if task and not task.is_running:
            task.start()
            return ControlTaskResponse(
                resultCode="200",
                resultValue=None,
                resultHint="任务已启动",
            )
        elif task and task.is_running:
            return ControlTaskResponse(
                resultCode="200",
                resultValue=None,
                resultHint="任务已在运行中",
            )
        return ControlTaskResponse(
            resultCode="404",
            resultValue=None,
            resultHint="未找到指定任务",
        )

    elif req.command == 2:
        # 删除
        success = video_task_manager.delete_task(req.analyseId)
        if success:
            return ControlTaskResponse(
                resultCode="200",
                resultValue=None,
                resultHint="任务已删除",
            )
        return ControlTaskResponse(
            resultCode="404",
            resultValue=None,
            resultHint="未找到指定任务",
        )

    return ControlTaskResponse(
        resultCode="400",
        resultValue=None,
        resultHint="无效的控制指令",
    )
