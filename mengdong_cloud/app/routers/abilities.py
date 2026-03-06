# 4.1.7.1 算法能力获取接口

import logging

from fastapi import APIRouter

from app.config import MODEL_CONFIGS
from app.schemas import AbilitiesResponse, AbilityItem, AlgParam

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/service/abilities", response_model=AbilitiesResponse)
async def get_abilities():
    """获取服务平台算法能力列表"""
    abilities = []
    for _model_file, cfg in MODEL_CONFIGS.items():
        abilities.append(
            AbilityItem(
                algCode=cfg["algCode"],
                algDesc=cfg["algDesc"],
                algParams=[
                    AlgParam(key="--sensitivity", value="灵敏度,范围[1,5]")
                ],
            )
        )

    return AbilitiesResponse(
        resultCode="200",
        resultValue={
            "abilityInfo": {
                "number": len(abilities),
                "ability": [a.model_dump() for a in abilities],
            }
        },
        resultHint=None,
    )
