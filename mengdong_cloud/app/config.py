# 配置文件

import os

# 服务配置
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "22266"))

# 上报平台地址（统一视频平台）
PLATFORM_HOST = os.getenv("PLATFORM_HOST", "http://127.0.0.1:8080")

# 保活间隔（秒）
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "30"))

# 模型路径
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

# 输出目录
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")

# 推理置信度阈值
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# 推理设备
DEVICE = os.getenv("DEVICE", "0")  # GPU设备号，"cpu" 表示使用CPU

# 模型配置：模型文件名 -> (algCode, 描述, 标签列表)
MODEL_CONFIGS = {
    "model_mengdong_raa_adjusted.pt": {
        "algCode": "010101",
        "algDesc": "人员穿戴检测（安全帽、安全带）",
        "labels": {
            0: "person",
            1: "safetybelt",
            2: "mapblu",
            3: "mapor",
            4: "mapr",
            5: "mapwh",
            6: "mapy",
        },
        "labels_cn": {
            0: "人",
            1: "安全带",
            2: "蓝色安全帽",
            3: "橘色安全帽",
            4: "红色安全帽",
            5: "白色安全帽",
            6: "黄色安全帽",
        },
    },
    "model_mengdong_small_SRL.pt": {
        "algCode": "010102",
        "algDesc": "钩子与差速器检测",
        "labels": {
            0: "hook",
            1: "SRL",
        },
        "labels_cn": {
            0: "钩子",
            1: "差速器",
        },
    },
    "model_mengdong_tower.pt": {
        "algCode": "010103",
        "algDesc": "铁塔检测",
        "labels": {
            0: "tower",
        },
        "labels_cn": {
            0: "铁塔",
        },
    },
    "model_mengdong_scene_album.pt": {
        "algCode": "010104",
        "algDesc": "场景设备检测",
        "labels": {
            0: "AerialWorkBucketTruck",
            1: "ConductorSpacer",
            2: "byq",
            3: "daozha",
            4: "dlq",
            5: "gk",
            6: "jsj",
            7: "pcs",
            8: "pole",
            9: "xl",
        },
        "labels_cn": {
            0: "斗臂车",
            1: "导线间隔棒",
            2: "变压器",
            3: "刀闸",
            4: "断路器",
            5: "高空屋顶",
            6: "脚手架",
            7: "脚扣",
            8: "电线杆",
            9: "线路",
        },
    },
}

# algCode 到模型文件的反向映射
ALGCODE_TO_MODEL = {}
for model_file, cfg in MODEL_CONFIGS.items():
    ALGCODE_TO_MODEL[cfg["algCode"]] = model_file
