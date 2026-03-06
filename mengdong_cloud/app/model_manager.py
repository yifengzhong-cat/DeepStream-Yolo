# YOLOv8 模型管理器

import logging
import os
from typing import Dict, Optional

from app.config import CONFIDENCE_THRESHOLD, DEVICE, MODEL_CONFIGS, MODEL_DIR

logger = logging.getLogger(__name__)


class ModelManager:
    """管理多个 YOLOv8 模型的加载和推理"""

    def __init__(self):
        self._models: Dict[str, object] = {}
        self._loaded = False

    def load_all_models(self):
        """加载所有配置的模型"""
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

        self._loaded = True
        logger.info("模型加载完成，共加载 %d 个模型", len(self._models))

    def get_model(self, alg_code: str) -> Optional[object]:
        """根据算法编码获取模型"""
        return self._models.get(alg_code)

    def predict(self, alg_code: str, source, conf: float = None, **kwargs):
        """执行推理

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
    def is_loaded(self) -> bool:
        return self._loaded


# 全局单例
model_manager = ModelManager()
