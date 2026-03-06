from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelAbility:
    alg_code: str
    alg_desc: str
    infer_config: str
    labels: list[str]


REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_ABILITIES: dict[str, ModelAbility] = {
    "101001": ModelAbility(
        alg_code="101001",
        alg_desc="人员穿戴识别（安全带+安全帽）",
        infer_config=str(REPO_ROOT / "config_infer_primary_yoloV8.txt"),
        labels=["person", "safetybelt", "mapblu", "mapor", "mapr", "mapwh", "mapy"],
    ),
    "101002": ModelAbility(
        alg_code="101002",
        alg_desc="钩子/差速器识别（SRL）",
        infer_config=str(REPO_ROOT / "config_infer_primary_yoloV8.txt"),
        labels=["hook", "SRL"],
    ),
    "101003": ModelAbility(
        alg_code="101003",
        alg_desc="铁塔识别",
        infer_config=str(REPO_ROOT / "config_infer_primary_yoloV8.txt"),
        labels=["tower"],
    ),
    "101004": ModelAbility(
        alg_code="101004",
        alg_desc="场景相册识别",
        infer_config=str(REPO_ROOT / "config_infer_primary_yoloV8.txt"),
        labels=[
            "AerialWorkBucketTruck",
            "ConductorSpacer",
            "byq",
            "daozha",
            "dlq",
            "gk",
            "jsj",
            "pcs",
            "pole",
            "xl",
        ],
    ),
}


def get_ability_or_none(alg_code: str) -> ModelAbility | None:
    return MODEL_ABILITIES.get(alg_code)
