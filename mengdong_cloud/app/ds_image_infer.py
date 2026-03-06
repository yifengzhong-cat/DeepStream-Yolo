# DeepStream 图片推理模块
#
# 使用 DeepStream GStreamer 管道对单张图片进行推理，
# 与视频流推理共用相同的 nvinfer 配置（config_infer_*.txt）和
# 自定义解析库（libnvdsinfer_custom_impl_Yolo.so）。
#
# 管道结构：
#   filesrc → jpegdec → videoconvert → nvvideoconvert → capsfilter(NV12/NVMM)
#     → streammux → nvinfer → fakesink
#                     ↑
#               probe 提取 NvDsObjectMeta 检测结果

import logging
import os
import tempfile
from typing import List

import cv2

logger = logging.getLogger(__name__)

# 检测 DeepStream 是否可用
_DEEPSTREAM_AVAILABLE = False
try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # noqa: F401
    import pyds  # noqa: F401

    _DEEPSTREAM_AVAILABLE = True
except (ImportError, ValueError):
    pass


def is_available() -> bool:
    """检查 DeepStream 图片推理是否可用"""
    return _DEEPSTREAM_AVAILABLE


def infer_image(image, infer_config_path: str) -> list:
    """使用 DeepStream 管道对单张图片进行推理

    通过构建一次性 GStreamer 管道处理单帧图片，使用与视频推理
    相同的 nvinfer 配置和自定义解析库。

    Args:
        image: BGR numpy 数组
        infer_config_path: nvinfer 推理配置文件路径
            (如 config_infer_010101.txt)

    Returns:
        Detection 对象列表
    """
    if not _DEEPSTREAM_AVAILABLE:
        raise RuntimeError(
            "DeepStream Python 绑定 (pyds/Gst) 不可用，无法进行图片推理"
        )

    from app.inference import Detection

    Gst.init(None)

    h, w = image.shape[:2]
    detections: List[Detection] = []

    # 将图片保存为临时 JPEG 文件
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(tmp_path, image)

    try:
        pipeline = Gst.Pipeline.new("image-infer")

        # 源：读取临时 JPEG 文件
        filesrc = Gst.ElementFactory.make("filesrc", "source")
        filesrc.set_property("location", tmp_path)

        jpegdec = Gst.ElementFactory.make("jpegdec", "decoder")

        # CPU 格式转换
        videoconvert = Gst.ElementFactory.make("videoconvert", "vidconv")

        # CPU → GPU 内存转换
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")

        # 指定 GPU 内存格式（nvinfer 输入要求）
        capsfilter = Gst.ElementFactory.make("capsfilter", "caps")
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"),
        )

        # 流复用器（nvinfer 前置要求）
        streammux = Gst.ElementFactory.make("nvstreammux", "mux")
        streammux.set_property("batch-size", 1)
        streammux.set_property("width", w)
        streammux.set_property("height", h)
        streammux.set_property("batched-push-timeout", 4000000)
        streammux.set_property("live-source", 0)

        # 主推理引擎
        nvinfer_elem = Gst.ElementFactory.make("nvinfer", "inference")
        nvinfer_elem.set_property("config-file-path", infer_config_path)

        # 丢弃输出（结果通过 probe 回调提取）
        fakesink = Gst.ElementFactory.make("fakesink", "sink")
        fakesink.set_property("sync", False)

        # 添加元素到管道
        for elem in [
            filesrc, jpegdec, videoconvert, nvvidconv,
            capsfilter, streammux, nvinfer_elem, fakesink,
        ]:
            if elem is None:
                raise RuntimeError(
                    "无法创建 GStreamer 元素，请检查 DeepStream 安装"
                )
            pipeline.add(elem)

        # 链接：filesrc → jpegdec → videoconvert → nvvidconv → capsfilter
        filesrc.link(jpegdec)
        jpegdec.link(videoconvert)
        videoconvert.link(nvvidconv)
        nvvidconv.link(capsfilter)

        # capsfilter → streammux（需要请求 sink pad）
        sink_pad = streammux.request_pad_simple("sink_0")
        src_pad = capsfilter.get_static_pad("src")
        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("无法连接 capsfilter → streammux")

        # streammux → nvinfer → fakesink
        streammux.link(nvinfer_elem)
        nvinfer_elem.link(fakesink)

        # 在 nvinfer src pad 添加 probe 提取检测结果
        def _probe_callback(pad, info):
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if batch_meta is None:
                return Gst.PadProbeReturn.OK

            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    rect = obj_meta.rect_params
                    detections.append(
                        Detection(
                            class_id=obj_meta.class_id,
                            confidence=obj_meta.confidence,
                            x1=int(rect.left),
                            y1=int(rect.top),
                            x2=int(rect.left + rect.width),
                            y2=int(rect.top + rect.height),
                        )
                    )
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

            return Gst.PadProbeReturn.OK

        nvinfer_src = nvinfer_elem.get_static_pad("src")
        nvinfer_src.add_probe(Gst.PadProbeType.BUFFER, _probe_callback)

        # 运行管道直到 EOS
        pipeline.set_state(Gst.State.PLAYING)

        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(
            30 * Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR,
        )

        if msg is not None and msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.error(
                "DeepStream 图片推理管道错误: %s: %s", err.message, dbg
            )

        pipeline.set_state(Gst.State.NULL)

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    logger.debug("DeepStream 图片推理完成，检测到 %d 个目标", len(detections))
    return detections
