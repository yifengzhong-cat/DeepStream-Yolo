# 4.1.7.5 图片分析任务接口

import logging

from fastapi import APIRouter

from app.inference import (
    decode_base64_image,
    draw_detections,
    encode_image_to_base64,
    get_current_time_str,
    get_sensitivity_from_rule,
    parse_results,
)
from app.model_manager import model_manager
from app.schemas import ImageTaskRequest, ImageTaskResponse, ImageTaskResultValue

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/service/imageTask", response_model=ImageTaskResponse)
async def image_task(req: ImageTaskRequest):
    """图片分析任务：对单张图片进行推理分析"""
    try:
        # 解码图片
        img = decode_base64_image(req.imageData)
    except Exception:
        logger.exception("图片解码失败")
        return ImageTaskResponse(
            resultCode="400",
            resultValue=None,
            resultHint="图片数据解码失败",
        )

    # 检查模型是否存在
    model = model_manager.get_model(req.algCode)
    if model is None:
        return ImageTaskResponse(
            resultCode="404",
            resultValue=None,
            resultHint=f"未找到算法编码对应的模型: {req.algCode}",
        )

    # 从规则中提取灵敏度
    conf = get_sensitivity_from_rule(req.rule)

    try:
        # 执行推理
        kwargs = {}
        if conf is not None:
            kwargs["conf"] = conf
        results = model_manager.predict(req.algCode, img, **kwargs)
    except Exception:
        logger.exception("推理失败")
        return ImageTaskResponse(
            resultCode="500",
            resultValue=None,
            resultHint="推理执行失败",
        )

    # 解析结果
    analyse_results, result_detail = parse_results(results, req.algCode)

    # 绘制检测框
    osd_img = draw_detections(img, results, req.algCode)

    # 编码图片
    raw_b64 = encode_image_to_base64(img)
    osd_b64 = encode_image_to_base64(osd_img)

    result_value = ImageTaskResultValue(
        analyseResults=analyse_results,
        analyseTime=get_current_time_str(),
        rawImageName=f"{req.analyseId}_raw.jpg",
        rawImageData=raw_b64,
        osdImageName=f"{req.analyseId}_osd.jpg",
        osdImageData=osd_b64,
        resultDetail=result_detail,
    )

    return ImageTaskResponse(
        resultCode="200",
        resultValue=result_value,
        resultHint=None,
    )
