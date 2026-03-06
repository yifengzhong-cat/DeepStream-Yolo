# 视频流处理任务管理器

import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, Optional

import cv2
import requests

from app.config import ALGCODE_TO_MODEL, MODEL_CONFIGS, OUTPUT_DIR, PLATFORM_HOST
from app.inference import (
    draw_detections,
    encode_image_to_base64,
    get_current_time_str,
    parse_results,
)
from app.model_manager import model_manager

logger = logging.getLogger(__name__)


class VideoTask:
    """单个视频分析任务"""

    def __init__(
        self,
        analyse_id: str,
        alg_code: str,
        video_url: str,
        dev_code: str = "",
        interval: int = 60,
        format_type: int = 0,
        rule=None,
    ):
        self.analyse_id = analyse_id
        self.alg_code = alg_code
        self.video_url = video_url
        self.dev_code = dev_code
        self.interval = interval
        self.format_type = format_type
        self.rule = rule
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

    def start(self):
        """启动视频分析任务"""
        if self._running:
            logger.warning("任务 %s 已在运行中", self.analyse_id)
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True, name=f"task-{self.analyse_id}"
        )
        self._thread.start()
        logger.info("视频任务已启动: %s", self.analyse_id)

    def stop(self):
        """停止视频分析任务"""
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("视频任务已停止: %s", self.analyse_id)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def osd_video_url(self) -> str:
        """获取标注视频的URL"""
        return f"/output/{self.analyse_id}.mp4"

    def _process_loop(self):
        """视频处理主循环"""
        cap = None
        writer = None
        try:
            cap = cv2.VideoCapture(self.video_url)
            if not cap.isOpened():
                logger.error("无法打开视频流: %s", self.video_url)
                self._running = False
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建输出视频写入器
            output_path = os.path.join(OUTPUT_DIR, f"{self.analyse_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            last_report_time = 0
            frame_count = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # 视频结束或断流，尝试重连
                    logger.warning("视频流读取失败，尝试重连: %s", self.video_url)
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(self.video_url)
                    if not cap.isOpened():
                        logger.error("重连失败: %s", self.video_url)
                        break
                    continue

                frame_count += 1

                # 执行推理
                try:
                    results = model_manager.predict(self.alg_code, frame)
                except Exception:
                    logger.exception("推理失败")
                    continue

                # 绘制检测结果并写入输出视频
                osd_frame = draw_detections(frame, results, self.alg_code)
                if writer is not None:
                    writer.write(osd_frame)

                # 按间隔上报结果
                current_time = time.time()
                if current_time - last_report_time >= self.interval:
                    last_report_time = current_time
                    self._report_result(frame, osd_frame, results)

        except Exception:
            logger.exception("视频处理异常: %s", self.analyse_id)
        finally:
            self._running = False
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            logger.info("视频处理结束: %s", self.analyse_id)

    def _report_result(self, raw_frame, osd_frame, results):
        """向统一视频平台上报分析结果"""
        try:
            analyse_results, result_detail = parse_results(results, self.alg_code)

            raw_b64 = encode_image_to_base64(raw_frame)
            osd_b64 = encode_image_to_base64(osd_frame)

            payload = {
                "algCode": self.alg_code,
                "analyseId": self.analyse_id,
                "analyseTime": get_current_time_str(),
                "analyseResults": analyse_results,
                "rawImageName": f"{self.analyse_id}_raw.jpg",
                "rawImageData": raw_b64,
                "osdImageName": f"{self.analyse_id}_osd.jpg",
                "osdImageData": osd_b64,
                "resultDetail": [d.model_dump() for d in result_detail],
            }

            url = f"{PLATFORM_HOST}/analysis/api/v1/analyseResult"
            resp = requests.post(url, json=payload, timeout=10)
            logger.info(
                "结果上报: analyseId=%s, status=%s", self.analyse_id, resp.status_code
            )
        except Exception:
            logger.exception("结果上报失败: %s", self.analyse_id)


class VideoTaskManager:
    """视频任务管理器"""

    def __init__(self):
        self._tasks: Dict[str, VideoTask] = {}
        self._lock = threading.Lock()

    def create_task(
        self,
        analyse_id: str,
        alg_code: str,
        video_url: str,
        dev_code: str = "",
        interval: int = 60,
        format_type: int = 0,
        rule=None,
    ) -> VideoTask:
        """创建并启动视频任务"""
        with self._lock:
            # 如果存在同名任务先停止
            if analyse_id in self._tasks:
                self._tasks[analyse_id].stop()

            task = VideoTask(
                analyse_id=analyse_id,
                alg_code=alg_code,
                video_url=video_url,
                dev_code=dev_code,
                interval=interval,
                format_type=format_type,
                rule=rule,
            )
            self._tasks[analyse_id] = task
            task.start()
            return task

    def stop_task(self, analyse_id: str) -> bool:
        """停止视频任务"""
        with self._lock:
            task = self._tasks.get(analyse_id)
            if task:
                task.stop()
                return True
            return False

    def delete_task(self, analyse_id: str) -> bool:
        """删除视频任务"""
        with self._lock:
            task = self._tasks.pop(analyse_id, None)
            if task:
                task.stop()
                return True
            return False

    def get_task(self, analyse_id: str) -> Optional[VideoTask]:
        """获取视频任务"""
        return self._tasks.get(analyse_id)

    def update_analyse_id(self, old_id: str, new_id: str) -> bool:
        """更新分析ID"""
        with self._lock:
            task = self._tasks.pop(old_id, None)
            if task is None:
                return False
            task.stop()
            # 使用新的 analyse_id 重新创建任务
            new_task = VideoTask(
                analyse_id=new_id,
                alg_code=task.alg_code,
                video_url=task.video_url.replace(old_id, new_id),
                dev_code=task.dev_code,
                interval=task.interval,
                format_type=task.format_type,
                rule=task.rule,
            )
            self._tasks[new_id] = new_task
            new_task.start()
            return True

    @property
    def active_tasks(self) -> Dict[str, VideoTask]:
        return {k: v for k, v in self._tasks.items() if v.is_running}


# 全局单例
video_task_manager = VideoTaskManager()
