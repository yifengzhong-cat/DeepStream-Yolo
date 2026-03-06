# 推理辅助函数

import base64
import io
import logging
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config import ALGCODE_TO_MODEL, MODEL_CONFIGS
from app.schemas import ResultDetail, ResultItem

logger = logging.getLogger(__name__)


def decode_base64_image(image_data: str) -> np.ndarray:
    """将 Base64 编码的图片解码为 numpy 数组 (BGR)"""
    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片数据")
    return img


def encode_image_to_base64(img: np.ndarray, fmt: str = ".jpg") -> str:
    """将 numpy 图片数组编码为 Base64 字符串"""
    _, buffer = cv2.imencode(fmt, img)
    return base64.b64encode(buffer).decode("utf-8")


def draw_detections(img: np.ndarray, results, alg_code: str) -> np.ndarray:
    """在图片上绘制检测框和标签，返回标注后的图片"""
    osd_img = img.copy()
    model_file = ALGCODE_TO_MODEL.get(alg_code)
    if model_file is None:
        return osd_img

    labels_cn = MODEL_CONFIGS[model_file].get("labels_cn", {})

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = labels_cn.get(cls_id, str(cls_id))

            # 绘制矩形框
            color = (0, 255, 0)
            cv2.rectangle(osd_img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            text = f"{label} {conf:.1%}"
            font_scale = 0.6
            thickness = 1
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(osd_img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                osd_img,
                text,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
            )

    return osd_img


def parse_results(results, alg_code: str) -> Tuple[List[str], List[ResultDetail]]:
    """解析 YOLO 推理结果为接口返回格式

    Returns:
        (analyseResults, resultDetail)
    """
    model_file = ALGCODE_TO_MODEL.get(alg_code)
    if model_file is None:
        return [], []

    labels_cn = MODEL_CONFIGS[model_file].get("labels_cn", {})

    # 按类别分组
    class_items = {}
    analyse_results_set = set()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
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

    analyse_results = list(analyse_results_set)

    result_details = []
    for cls_id, items in class_items.items():
        label = labels_cn.get(cls_id, str(cls_id))
        result_details.append(
            ResultDetail(
                algCode=alg_code,
                resultDesc=label,
                num=len(items),
                resultItems=items,
            )
        )

    return analyse_results, result_details


def get_sensitivity_from_rule(rule) -> float:
    """从 rule 参数中提取灵敏度设置，转化为置信度阈值"""
    if rule is None:
        return None
    for param in rule.algParams or []:
        if param.key == "--sensitivity":
            try:
                sensitivity = int(param.value)
                # 灵敏度 1-5 映射到置信度 0.8-0.2
                conf = max(0.1, 0.95 - sensitivity * 0.15)
                return conf
            except (ValueError, TypeError):
                pass
    return None


def get_current_time_str() -> str:
    """获取当前时间字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
