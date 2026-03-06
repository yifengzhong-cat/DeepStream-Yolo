# YOLOv8 模型管理器
#
# 使用 DeepStream-Yolo 推理管道作为主要推理后端：
# 1. 视频流：使用 DeepStream GStreamer 管道 (nvinfer + NvDsInferParseYolo)
# 2. 图片：使用 DeepStream 单帧推理管道 (ds_image_infer)
#
# 启动时通过 export_yoloV8.py 将 .pt 模型导出为 ONNX，
# DeepStream 的 nvinfer 自动将 ONNX 转为 TensorRT engine。
#
# 注意：ONNX 导出脚本 (export_yoloV8.py) 需要 ultralytics 库，
# 但运行时推理不需要。可提前导出 ONNX 模型后部署。

import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional

from app.config import (
    ALGCODE_TO_MODEL,
    DEEPSTREAM_INFER_SIZE,
    DEEPSTREAM_YOLO_DIR,
    MODEL_CONFIGS,
    MODEL_DIR,
)
from app.inference import Detection

logger = logging.getLogger(__name__)


class ModelManager:
    """管理 YOLOv8 模型的加载和推理（基于 DeepStream-Yolo）"""

    def __init__(self):
        self._loaded = False
        self._ds_configs: Dict[str, dict] = {}
        self._ds_image_available = False

    def load_all_models(self):
        """加载所有配置的模型

        流程：
        1. 尝试将 .pt 模型通过 export_yoloV8.py 导出为 ONNX（需要 ultralytics）
        2. 生成 DeepStream 推理配置文件
        3. 检查 DeepStream 图片推理是否可用
        """
        # 步骤1：尝试导出 ONNX
        self._export_onnx_models()

        # 步骤2：生成 DeepStream 配置文件
        self._generate_ds_configs()

        # 步骤3：检查 DeepStream 图片推理可用性
        self._check_ds_image_infer()

        self._loaded = True
        logger.info(
            "模型加载完成，DeepStream 配置 %d 个，图片推理: %s",
            len(self._ds_configs),
            "DeepStream" if self._ds_image_available else "不可用",
        )

    def _export_onnx_models(self):
        """使用 DeepStream-Yolo 的 export_yoloV8.py 将 .pt 模型导出为 .onnx

        注意：此脚本内部使用 ultralytics 库加载 .pt 模型，
        仅在 ONNX 文件不存在时才需要运行。如果 ONNX 已导出则跳过。
        """
        export_script = os.path.join(DEEPSTREAM_YOLO_DIR, "utils", "export_yoloV8.py")
        if not os.path.exists(export_script):
            logger.warning(
                "DeepStream-Yolo 导出脚本不存在: %s，跳过 ONNX 导出", export_script
            )
            return

        for model_file, cfg in MODEL_CONFIGS.items():
            pt_path = os.path.join(MODEL_DIR, model_file)
            onnx_file = model_file.rsplit(".", 1)[0] + ".onnx"
            onnx_path = os.path.join(MODEL_DIR, onnx_file)

            if not os.path.exists(pt_path):
                logger.warning("模型文件不存在，跳过导出: %s", pt_path)
                continue

            if os.path.exists(onnx_path):
                logger.info("ONNX 模型已存在，跳过导出: %s", onnx_path)
                continue

            logger.info(
                "正在导出 ONNX 模型: %s -> %s (使用 DeepStream-Yolo export_yoloV8.py)",
                model_file,
                onnx_file,
            )
            try:
                # --dynamic: 启用动态 batch size，DeepStream 可按需调整
                # --simplify: 优化 ONNX 计算图，提升 TensorRT 转换效率
                cmd = [
                    sys.executable,
                    export_script,
                    "-w", pt_path,
                    "-s", str(DEEPSTREAM_INFER_SIZE),
                    "--dynamic",
                    "--simplify",
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=MODEL_DIR,
                )
                if result.returncode == 0:
                    logger.info("ONNX 导出成功: %s", onnx_path)
                else:
                    logger.error(
                        "ONNX 导出失败: %s\nstdout: %s\nstderr: %s",
                        model_file,
                        result.stdout,
                        result.stderr,
                    )
            except Exception:
                logger.exception("ONNX 导出异常: %s", model_file)

    def _generate_ds_configs(self):
        """生成 DeepStream 配置文件"""
        try:
            from app.deepstream_config import generate_all_configs

            self._ds_configs = generate_all_configs()
            if self._ds_configs:
                logger.info("DeepStream 配置文件生成成功")
        except Exception:
            logger.exception("DeepStream 配置文件生成失败")

    def _check_ds_image_infer(self):
        """检查 DeepStream 图片推理是否可用"""
        try:
            from app import ds_image_infer

            self._ds_image_available = ds_image_infer.is_available()
            if self._ds_image_available:
                logger.info("DeepStream 图片推理可用")
            else:
                logger.warning(
                    "DeepStream 图片推理不可用（pyds/Gst 未安装），"
                    "图片分析接口将不可用"
                )
        except ImportError:
            logger.warning("ds_image_infer 模块导入失败")
            self._ds_image_available = False

    def get_ds_config(self, alg_code: str) -> Optional[dict]:
        """根据算法编码获取 DeepStream 配置"""
        return self._ds_configs.get(alg_code)

    def get_onnx_path(self, alg_code: str) -> Optional[str]:
        """根据算法编码获取 ONNX 模型路径"""
        model_file = ALGCODE_TO_MODEL.get(alg_code)
        if model_file is None:
            return None
        onnx_file = model_file.rsplit(".", 1)[0] + ".onnx"
        onnx_path = os.path.join(MODEL_DIR, onnx_file)
        if os.path.exists(onnx_path):
            return onnx_path
        return None

    def predict(self, alg_code: str, source, conf: float = None, **kwargs) -> List[Detection]:
        """对图片执行推理，返回检测结果列表

        使用 DeepStream 管道进行推理（与视频流推理共用同一套配置）。

        Args:
            alg_code: 算法编码
            source: BGR numpy 数组
            conf: 置信度阈值（由 DeepStream 配置中的 pre-cluster-threshold 控制）

        Returns:
            Detection 对象列表
        """
        if self._ds_image_available:
            ds_cfg = self.get_ds_config(alg_code)
            if ds_cfg is not None:
                from app import ds_image_infer

                return ds_image_infer.infer_image(
                    source, ds_cfg["infer_config_path"]
                )

        raise RuntimeError(
            f"无可用推理后端: algCode={alg_code}。"
            "请确保 DeepStream (pyds/Gst) 已安装且 ONNX 模型已导出。"
        )

    @property
    def loaded_models(self) -> Dict[str, dict]:
        """返回已加载的 DeepStream 配置（兼容旧接口）"""
        return self._ds_configs

    @property
    def ds_configs(self) -> Dict[str, dict]:
        return self._ds_configs

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# 全局单例
model_manager = ModelManager()
