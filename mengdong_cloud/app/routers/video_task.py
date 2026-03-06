# 4.1.7.2 视频任务管理接口

import logging

from fastapi import APIRouter

from app.config import PUBLIC_HOST, SERVICE_PORT
from app.schemas import VideoTaskRequest, VideoTaskResponse, VideoTaskResultItem
from app.tasks.video_processor import video_task_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/service/videoTask", response_model=VideoTaskResponse)
async def video_task(req: VideoTaskRequest):
    """视频任务管理：创建/启动/停止/删除视频分析任务"""
    result_items = []

    for vi in req.videoInfo:
        if req.command == 1:
            # 开始任务
            task = video_task_manager.create_task(
                analyse_id=vi.analyseId,
                alg_code=req.algCode,
                video_url=vi.videoUrl,
                dev_code=vi.devCode,
                interval=req.interval or 60,
                format_type=vi.formatType,
                rule=req.rule,
            )
            osd_url = (
                f"http://{PUBLIC_HOST}:{SERVICE_PORT}/output/{vi.analyseId}.mp4"
            )
            result_items.append(
                VideoTaskResultItem(
                    analyseId=vi.analyseId,
                    devCode=vi.devCode,
                    osdVideoUrl=osd_url,
                )
            )
        elif req.command == 0:
            # 停止任务
            video_task_manager.stop_task(vi.analyseId)
            result_items.append(
                VideoTaskResultItem(
                    analyseId=vi.analyseId,
                    devCode=vi.devCode,
                )
            )
        elif req.command == 2:
            # 删除任务
            video_task_manager.delete_task(vi.analyseId)
            result_items.append(
                VideoTaskResultItem(
                    analyseId=vi.analyseId,
                    devCode=vi.devCode,
                )
            )

    return VideoTaskResponse(
        resultCode="200",
        resultValue=result_items,
        resultHint=None,
    )
