# 视频流处理任务管理器
#
# 使用 DeepStream-Yolo 的 GStreamer 推理管道进行视频流分析。
# DeepStream 管道结构（参照 deepstream_app_config.txt）：
#
#   uridecodebin → streammux → nvinfer(GIE) → nvvideoconvert → nvosd
#     → nvvideoconvert → video/x-raw → appsink
#
# nvinfer 使用生成的 config_infer_<algCode>.txt 配置文件，内部加载：
# - ONNX 模型（由 export_yoloV8.py 从 .pt 导出）
# - 自定义解析库 libnvdsinfer_custom_impl_Yolo.so（NvDsInferParseYolo）
# - TensorRT 引擎（首次运行自动从 ONNX 构建）

import logging
import os
import threading
import time
from typing import Dict, Optional

import cv2
import requests

from app.config import (
    ALGCODE_TO_MODEL,
    DEEPSTREAM_CONFIG_DIR,
    DEEPSTREAM_STREAMMUX_BATCH_TIMEOUT,
    DEEPSTREAM_STREAMMUX_HEIGHT,
    DEEPSTREAM_STREAMMUX_WIDTH,
    MODEL_CONFIGS,
    OUTPUT_DIR,
    PLATFORM_HOST,
)
from app.inference import (
    encode_image_to_base64,
    get_current_time_str,
)
from app.model_manager import model_manager

logger = logging.getLogger(__name__)

# 检测 DeepStream Python 绑定是否可用
_DEEPSTREAM_AVAILABLE = False
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    import pyds
    _DEEPSTREAM_AVAILABLE = True
    logger.info("DeepStream Python 绑定 (pyds) 可用，将使用 DeepStream 管道")
except (ImportError, ValueError):
    logger.warning(
        "DeepStream Python 绑定 (pyds) 不可用，视频分析功能将不可用"
    )


# ============================================================
# DeepStream 管道方式
# ============================================================

class DeepStreamVideoTask:
    """使用 DeepStream GStreamer 管道处理视频流

    管道参照 DeepStream-Yolo 项目的 deepstream_app_config.txt 构建：
    uridecodebin → streammux → nvinfer → nvvideoconvert → nvosd
      → nvvideoconvert → capsfilter(BGRx) → videoconvert → capsfilter(BGR)
      → appsink
    """

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
        self._pipeline = None
        self._loop = None
        self._last_report_time = 0

    def start(self):
        if self._running:
            logger.warning("任务 %s 已在运行中", self.analyse_id)
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_pipeline, daemon=True, name=f"ds-{self.analyse_id}"
        )
        self._thread.start()
        logger.info("DeepStream 视频任务已启动: %s", self.analyse_id)

    def stop(self):
        self._stop_event.set()
        self._running = False
        if self._loop is not None:
            self._loop.quit()
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
        logger.info("DeepStream 视频任务已停止: %s", self.analyse_id)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def osd_video_url(self) -> str:
        return f"/output/{self.analyse_id}.mp4"

    def _run_pipeline(self):
        """构建并运行 DeepStream GStreamer 管道"""
        try:
            Gst.init(None)

            # 获取推理配置
            ds_cfg = model_manager.get_ds_config(self.alg_code)
            if ds_cfg is None:
                logger.error("未找到 DeepStream 配置: algCode=%s", self.alg_code)
                self._running = False
                return

            infer_config_path = ds_cfg["infer_config_path"]
            output_path = os.path.join(OUTPUT_DIR, f"{self.analyse_id}.mp4")

            # 构建管道（参照 DeepStream-Yolo 的 deepstream_app_config.txt）
            pipeline = Gst.Pipeline()

            # 源：uridecodebin（支持 RTSP/HTTP/文件）
            source = Gst.ElementFactory.make("uridecodebin", "source")
            source.set_property("uri", self.video_url)

            # 流复用器 streammux（参照 [streammux] 配置）
            streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
            streammux.set_property("batch-size", 1)
            streammux.set_property("width", DEEPSTREAM_STREAMMUX_WIDTH)
            streammux.set_property("height", DEEPSTREAM_STREAMMUX_HEIGHT)
            streammux.set_property(
                "batched-push-timeout", DEEPSTREAM_STREAMMUX_BATCH_TIMEOUT
            )

            # 主推理引擎 nvinfer（参照 [primary-gie] 配置）
            # 使用 DeepStream-Yolo 的 config_infer_primary 格式配置
            pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
            pgie.set_property("config-file-path", infer_config_path)

            # 视频格式转换
            nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv1")

            # OSD 叠加显示（参照 [osd] 配置）
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            nvosd.set_property("process-mode", 0)

            # 转换为 CPU 内存格式用于 appsink 取帧
            nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv2")
            capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
            caps = Gst.Caps.from_string("video/x-raw, format=BGRx")
            capsfilter.set_property("caps", caps)

            videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
            capsfilter2 = Gst.ElementFactory.make("capsfilter", "capsfilter2")
            caps2 = Gst.Caps.from_string("video/x-raw, format=BGR")
            capsfilter2.set_property("caps", caps2)

            # 输出 appsink（获取帧数据用于写入视频和截图上报）
            appsink = Gst.ElementFactory.make("appsink", "appsink")
            appsink.set_property("emit-signals", True)
            appsink.set_property("max-buffers", 1)
            appsink.set_property("drop", True)

            # 添加元素到管道
            for elem in [
                source, streammux, pgie, nvvidconv1, nvosd,
                nvvidconv2, capsfilter, videoconvert, capsfilter2, appsink,
            ]:
                pipeline.add(elem)

            # 链接管道（source 通过 pad-added 动态链接）
            streammux.link(pgie)
            pgie.link(nvvidconv1)
            nvvidconv1.link(nvosd)
            nvosd.link(nvvidconv2)
            nvvidconv2.link(capsfilter)
            capsfilter.link(videoconvert)
            videoconvert.link(capsfilter2)
            capsfilter2.link(appsink)

            # source pad-added 回调
            def on_pad_added(src, pad):
                sink_pad = streammux.request_pad_simple("sink_0")
                if sink_pad and not sink_pad.is_linked():
                    pad.link(sink_pad)

            source.connect("pad-added", on_pad_added)

            # OSD probe 回调：提取 DeepStream 检测元数据
            osd_sink_pad = nvosd.get_static_pad("sink")
            osd_sink_pad.add_probe(
                Gst.PadProbeType.BUFFER, self._osd_probe_callback
            )

            # appsink 回调：获取帧用于写入视频文件
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                (DEEPSTREAM_STREAMMUX_WIDTH, DEEPSTREAM_STREAMMUX_HEIGHT),
            )
            self._writer = writer

            def on_new_sample(sink):
                sample = sink.emit("pull-sample")
                if sample is None:
                    return Gst.FlowReturn.ERROR
                buf = sample.get_buffer()
                caps_s = sample.get_caps()
                info = caps_s.get_structure(0)
                w = info.get_int("width")[1]
                h = info.get_int("height")[1]
                ok, mapinfo = buf.map(Gst.MapFlags.READ)
                if ok:
                    import numpy as np
                    frame = np.ndarray(
                        shape=(h, w, 3), dtype=np.uint8, buffer=mapinfo.data
                    )
                    writer.write(frame)
                    buf.unmap(mapinfo)
                return Gst.FlowReturn.OK

            appsink.connect("new-sample", on_new_sample)

            self._pipeline = pipeline

            # 启动管道
            pipeline.set_state(Gst.State.PLAYING)

            # GLib 主循环
            self._loop = GLib.MainLoop()

            # 添加停止检查
            def check_stop():
                if self._stop_event.is_set():
                    self._loop.quit()
                    return False
                return True

            GLib.timeout_add(1000, check_stop)

            # 错误处理
            bus = pipeline.get_bus()
            bus.add_signal_watch()

            def on_message(bus, msg):
                t = msg.type
                if t == Gst.MessageType.EOS:
                    logger.info("视频流结束: %s", self.analyse_id)
                    self._loop.quit()
                elif t == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    logger.error(
                        "GStreamer 错误: %s: %s", err.message, dbg
                    )
                    self._loop.quit()

            bus.connect("message", on_message)

            self._loop.run()

        except Exception:
            logger.exception("DeepStream 管道异常: %s", self.analyse_id)
        finally:
            self._running = False
            if self._pipeline is not None:
                self._pipeline.set_state(Gst.State.NULL)
            if hasattr(self, "_writer") and self._writer is not None:
                self._writer.release()
            logger.info("DeepStream 视频处理结束: %s", self.analyse_id)

    def _osd_probe_callback(self, pad, info):
        """OSD sink pad probe 回调 — 提取 DeepStream 检测元数据

        DeepStream 的 nvinfer 引擎使用 NvDsInferParseYolo 解析 YOLO 输出，
        检测结果以 NvDsObjectMeta 形式附加到每帧的 batch_meta 中。
        """
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            # 按间隔上报结果
            current_time = time.time()
            if current_time - self._last_report_time >= self.interval:
                self._last_report_time = current_time
                self._report_ds_result(frame_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _report_ds_result(self, frame_meta):
        """从 DeepStream 帧元数据中提取检测结果并上报"""
        try:
            model_file = ALGCODE_TO_MODEL.get(self.alg_code)
            if model_file is None:
                return

            labels_cn = MODEL_CONFIGS[model_file].get("labels_cn", {})

            # 从 NvDsObjectMeta 提取检测框
            from app.schemas import ResultDetail, ResultItem

            class_items = {}
            analyse_results_set = set()

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                cls_id = obj_meta.class_id
                conf = obj_meta.confidence
                rect = obj_meta.rect_params

                x1 = int(rect.left)
                y1 = int(rect.top)
                x2 = int(rect.left + rect.width)
                y2 = int(rect.top + rect.height)

                label = labels_cn.get(cls_id, str(cls_id))
                analyse_results_set.add(label)

                if cls_id not in class_items:
                    class_items[cls_id] = []
                class_items[cls_id].append(
                    ResultItem(
                        score=round(conf * 100, 1),
                        leftTopX=x1,
                        leftTopY=y1,
                        rightBottomX=x2,
                        rightBottomY=y2,
                    )
                )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            result_details = []
            for cls_id, items in class_items.items():
                label = labels_cn.get(cls_id, str(cls_id))
                result_details.append(
                    ResultDetail(
                        algCode=self.alg_code,
                        resultDesc=label,
                        num=len(items),
                        resultItems=items,
                    )
                )

            analyse_results = list(analyse_results_set)

            payload = {
                "algCode": self.alg_code,
                "analyseId": self.analyse_id,
                "analyseTime": get_current_time_str(),
                "analyseResults": analyse_results,
                "rawImageName": f"{self.analyse_id}_raw.jpg",
                "rawImageData": "",
                "osdImageName": f"{self.analyse_id}_osd.jpg",
                "osdImageData": "",
                "resultDetail": [d.model_dump() for d in result_details],
            }

            url = f"{PLATFORM_HOST}/analysis/api/v1/analyseResult"
            resp = requests.post(url, json=payload, timeout=10)
            logger.info(
                "DeepStream 结果上报: analyseId=%s, status=%s",
                self.analyse_id,
                resp.status_code,
            )
        except Exception:
            logger.exception("DeepStream 结果上报失败: %s", self.analyse_id)


# ============================================================
# 统一 VideoTask 工厂 — 使用 DeepStream 管道
# ============================================================

def _create_video_task(
    analyse_id, alg_code, video_url, dev_code, interval, format_type, rule
):
    """创建视频处理任务（使用 DeepStream 管道）"""
    if not _DEEPSTREAM_AVAILABLE:
        raise RuntimeError(
            "DeepStream (pyds/Gst) 不可用，无法创建视频分析任务。"
            "请确保 DeepStream SDK 已安装。"
        )
    if not model_manager.get_ds_config(alg_code):
        raise RuntimeError(
            f"未找到 algCode={alg_code} 的 DeepStream 配置，"
            "请确保 ONNX 模型已导出且配置文件已生成。"
        )
    return DeepStreamVideoTask(
        analyse_id=analyse_id,
        alg_code=alg_code,
        video_url=video_url,
        dev_code=dev_code,
        interval=interval,
        format_type=format_type,
        rule=rule,
    )


class VideoTaskManager:
    """视频任务管理器"""

    def __init__(self):
        self._tasks: Dict[str, object] = {}
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
    ):
        """创建并启动视频任务"""
        with self._lock:
            # 如果存在同名任务先停止
            if analyse_id in self._tasks:
                self._tasks[analyse_id].stop()

            task = _create_video_task(
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

    def get_task(self, analyse_id: str):
        """获取视频任务"""
        return self._tasks.get(analyse_id)

    def update_analyse_id(self, old_id: str, new_id: str) -> bool:
        """更新分析ID"""
        with self._lock:
            task = self._tasks.pop(old_id, None)
            if task is None:
                return False
            task.stop()
            # 仅替换 URL 中查询参数部分的 analyseID 值
            new_url = task.video_url
            if f"analyseID={old_id}" in new_url:
                new_url = new_url.replace(f"analyseID={old_id}", f"analyseID={new_id}")
            elif f"analyseId={old_id}" in new_url:
                new_url = new_url.replace(f"analyseId={old_id}", f"analyseId={new_id}")
            new_task = _create_video_task(
                analyse_id=new_id,
                alg_code=task.alg_code,
                video_url=new_url,
                dev_code=task.dev_code,
                interval=task.interval,
                format_type=task.format_type,
                rule=task.rule,
            )
            self._tasks[new_id] = new_task
            new_task.start()
            return True

    @property
    def active_tasks(self) -> Dict[str, object]:
        return {k: v for k, v in self._tasks.items() if v.is_running}


# 全局单例
video_task_manager = VideoTaskManager()
