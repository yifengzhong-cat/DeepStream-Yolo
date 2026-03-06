# YOLOv8 模型管理器
#
# 支持两种推理后端：
# 1. DeepStream (TensorRT)：使用 DeepStream-Yolo 项目的 export_yoloV8.py 将 .pt
#    导出为 .onnx，再通过 DeepStream 的 GIE 引擎自动转为 TensorRT engine。
#    适合视频流分析场景，性能更高。
# 2. ultralytics（回退）：直接使用 ultralytics YOLO 加载 .pt 模型进行推理。
#    适合图片推理场景或 DeepStream 不可用的环境。

import logging
import os
import subprocess
import sys
from typing import Dict, Optional

from app.config import (
    ALGCODE_TO_MODEL,
    CONFIDENCE_THRESHOLD,
    DEEPSTREAM_INFER_SIZE,
    DEEPSTREAM_YOLO_DIR,
    DEVICE,
    MODEL_CONFIGS,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """管理多个 YOLOv8 模型的加载和推理"""

    def __init__(self):
        self._models: Dict[str, object] = {}
        self._loaded = False
        self._ds_configs: Dict[str, dict] = {}
        self._deepstream_available = False

    def load_all_models(self):
        """加载所有配置的模型

        流程：
        1. 尝试将 .pt 模型通过 DeepStream-Yolo 的 export_yoloV8.py 导出为 .onnx
        2. 生成 DeepStream 推理配置文件
        3. 加载 ultralytics YOLO 模型作为图片推理后端
        """
        # 步骤1：尝试导出 ONNX（使用 DeepStream-Yolo 的 export 脚本）
        self._export_onnx_models()

        # 步骤2：生成 DeepStream 配置文件
        self._generate_ds_configs()

        # 步骤3：加载 ultralytics 模型（用于图片推理）
        self._load_ultralytics_models()

        self._loaded = True
        logger.info("模型加载完成，共加载 %d 个模型", len(self._models))

    def _export_onnx_models(self):
        """使用 DeepStream-Yolo 的 export_yoloV8.py 将 .pt 模型导出为 .onnx"""
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

    def _load_ultralytics_models(self):
        """加载 ultralytics YOLO 模型（用于图片推理后端）"""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics 库未安装，请执行: pip install ultralytics")
            raise

        for model_file, cfg in MODEL_CONFIGS.items():
            model_path = os.path.join(MODEL_DIR, model_file)
            if os.path.exists(model_path):
                try:
                    model = YOLO(model_path)
                    self._models[cfg["algCode"]] = model
                    logger.info(
                        "模型加载成功: %s (algCode=%s, desc=%s)",
                        model_file,
                        cfg["algCode"],
                        cfg["algDesc"],
                    )
                except Exception:
                    logger.exception("模型加载失败: %s", model_file)
            else:
                logger.warning("模型文件不存在: %s", model_path)

    def get_model(self, alg_code: str) -> Optional[object]:
        """根据算法编码获取 ultralytics 模型"""
        return self._models.get(alg_code)

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

    def predict(self, alg_code: str, source, conf: float = None, **kwargs):
        """使用 ultralytics 后端执行推理（图片分析）

        Args:
            alg_code: 算法编码
            source: 图片路径、numpy数组或URL
            conf: 置信度阈值
            **kwargs: 其他 YOLO predict 参数

        Returns:
            YOLO 推理结果列表
        """
        model = self.get_model(alg_code)
        if model is None:
            raise ValueError(f"未找到算法编码对应的模型: {alg_code}")

        if conf is None:
            conf = CONFIDENCE_THRESHOLD

        device = DEVICE
        results = model.predict(source=source, conf=conf, device=device, **kwargs)
        return results

    @property
    def loaded_models(self) -> Dict[str, object]:
        return self._models

    @property
    def ds_configs(self) -> Dict[str, dict]:
        return self._ds_configs

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# 全局单例
model_manager = ModelManager()
