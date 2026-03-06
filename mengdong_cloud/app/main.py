# 蒙东云端AI分析平台 - 主入口
#
# 基于 DeepStream-Yolo 推理管道，所有推理均通过 DeepStream 完成：
# 1. 启动时使用 export_yoloV8.py 将 .pt 模型导出为 ONNX
# 2. 生成 DeepStream 推理配置文件（config_infer_*.txt, labels_*.txt）
# 3. 视频流分析使用 DeepStream GStreamer 管道（nvinfer + NvDsInferParseYolo）
# 4. 图片分析使用 DeepStream 单帧推理管道（ds_image_infer）
# 5. 不依赖 ultralytics 运行时库

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import (
    DEEPSTREAM_CONFIG_DIR,
    KEEPALIVE_INTERVAL,
    OUTPUT_DIR,
    PLATFORM_HOST,
    SERVICE_HOST,
    SERVICE_PORT,
)
from app.model_manager import model_manager
from app.routers import (
    abilities,
    control_task,
    image_task,
    keep_alive,
    update_analyse,
    upload_samples,
    video_task,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 保活线程停止事件
_keepalive_stop = threading.Event()


def _keepalive_loop():
    """定时发送保活报文"""
    url = f"{PLATFORM_HOST}/analysis/api/v1/keepAlive"
    while not _keepalive_stop.is_set():
        try:
            payload = {"devIP": SERVICE_HOST, "devPort": SERVICE_PORT}
            resp = requests.post(url, json=payload, timeout=5)
            logger.debug("保活上报: status=%s", resp.status_code)
        except Exception:
            logger.debug("保活上报失败（平台可能未就绪）")
        _keepalive_stop.wait(KEEPALIVE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 确保配置目录和输出目录存在
    os.makedirs(DEEPSTREAM_CONFIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 启动时加载模型（含 ONNX 导出和 DeepStream 配置生成）
    logger.info("正在加载 YOLOv8 模型（含 DeepStream-Yolo ONNX 导出）...")
    model_manager.load_all_models()
    logger.info("模型加载完成")

    # 启动保活线程
    _keepalive_stop.clear()
    keepalive_thread = threading.Thread(
        target=_keepalive_loop, daemon=True, name="keepalive"
    )
    keepalive_thread.start()
    logger.info("保活线程已启动")

    yield

    # 关闭时清理
    _keepalive_stop.set()
    logger.info("服务关闭")


app = FastAPI(
    title="蒙东云端AI分析平台",
    description=(
        "基于 DeepStream-Yolo + YOLOv8 的云端AI分析服务。\n\n"
        "所有推理均通过 NVIDIA DeepStream SDK 推理管道完成 "
        "(ONNX → TensorRT → nvinfer + NvDsInferParseYolo)，"
        "视频流和图片分析共用同一套 DeepStream 配置，无需 ultralytics 运行时依赖。"
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# 注册路由
app.include_router(abilities.router, tags=["算法能力"])
app.include_router(video_task.router, tags=["视频任务"])
app.include_router(control_task.router, tags=["任务控制"])
app.include_router(image_task.router, tags=["图片分析"])
app.include_router(keep_alive.router, tags=["保活"])
app.include_router(update_analyse.router, tags=["更新分析ID"])
app.include_router(upload_samples.router, tags=["样本上传"])

# 挂载输出目录为静态文件（提供视频下载）
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "models_loaded": model_manager.is_loaded,
        "loaded_count": len(model_manager.loaded_models),
        "deepstream_configs": len(model_manager.ds_configs),
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=False,
        workers=1,
    )
