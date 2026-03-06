# DeepStream 配置文件生成器
#
# 根据 MODEL_CONFIGS 为每个模型生成：
# 1. labels_<algCode>.txt         — 标签文件
# 2. config_infer_<algCode>.txt   — DeepStream 推理引擎配置
# 3. deepstream_app_<algCode>.txt — DeepStream 应用管道配置（用于视频流）
#
# 生成的配置文件格式与 DeepStream-Yolo 项目保持一致，使用相同的
# NvDsInferParseYolo 解析函数和 NvDsInferYoloCudaEngineGet 引擎创建函数。

import logging
import os

from app.config import (
    DEEPSTREAM_CONFIG_DIR,
    DEEPSTREAM_CUSTOM_LIB,
    DEEPSTREAM_INFER_SIZE,
    DEEPSTREAM_NETWORK_MODE,
    DEEPSTREAM_NMS_IOU_THRESHOLD,
    DEEPSTREAM_PRE_CLUSTER_THRESHOLD,
    DEEPSTREAM_STREAMMUX_BATCH_TIMEOUT,
    DEEPSTREAM_STREAMMUX_HEIGHT,
    DEEPSTREAM_STREAMMUX_WIDTH,
    DEEPSTREAM_TOPK,
    MODEL_CONFIGS,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)


def generate_labels_file(alg_code: str, labels: dict, output_dir: str) -> str:
    """生成 DeepStream 标签文件 (与 labels.txt 格式一致)

    每行一个类别名称，行号即类别索引。
    """
    labels_path = os.path.join(output_dir, f"labels_{alg_code}.txt")
    sorted_labels = [labels[i] for i in sorted(labels.keys())]
    with open(labels_path, "w", encoding="utf-8") as f:
        for name in sorted_labels:
            f.write(f"{name}\n")
    logger.info("标签文件已生成: %s", labels_path)
    return labels_path


def generate_infer_config(
    alg_code: str,
    onnx_path: str,
    labels_path: str,
    num_classes: int,
    output_dir: str,
) -> str:
    """生成 DeepStream 推理引擎配置文件

    格式参照 DeepStream-Yolo 项目的 config_infer_primary_yoloV8.txt，
    使用相同的 custom-lib-path / parse-bbox-func-name / engine-create-func-name。
    """
    config_path = os.path.join(output_dir, f"config_infer_{alg_code}.txt")
    engine_path = onnx_path.rsplit(".", 1)[0] + f"_b1_gpu0_fp{_mode_str()}.engine"

    content = f"""[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file={onnx_path}
model-engine-file={engine_path}
labelfile-path={labels_path}
batch-size=1
network-mode={DEEPSTREAM_NETWORK_MODE}
num-detected-classes={num_classes}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path={DEEPSTREAM_CUSTOM_LIB}
engine-create-func-name=NvDsInferYoloCudaEngineGet

[class-attrs-all]
nms-iou-threshold={DEEPSTREAM_NMS_IOU_THRESHOLD}
pre-cluster-threshold={DEEPSTREAM_PRE_CLUSTER_THRESHOLD}
topk={DEEPSTREAM_TOPK}
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("推理配置已生成: %s", config_path)
    return config_path


def generate_deepstream_app_config(
    alg_code: str,
    infer_config_path: str,
    source_uri: str,
    output_dir: str,
    output_file: str = "",
    live_source: bool = True,
) -> str:
    """生成 DeepStream 应用配置文件

    格式参照 DeepStream-Yolo 项目的 deepstream_app_config.txt。
    """
    app_config_path = os.path.join(
        output_dir, f"deepstream_app_{alg_code}.txt"
    )

    # 输出可选择文件或显示
    sink_section = ""
    if output_file:
        sink_section = f"""[sink0]
enable=1
type=3
container=1
codec=1
output-file={output_file}
sync=0
gpu-id=0
nvbuf-memory-type=0
"""
    else:
        sink_section = """[sink0]
enable=1
type=2
sync=0
gpu-id=0
nvbuf-memory-type=0
"""

    content = f"""[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[tiled-display]
enable=0
rows=1
columns=1
width=1280
height=720
gpu-id=0
nvbuf-memory-type=0

[source0]
enable=1
type=3
uri={source_uri}
num-sources=1
gpu-id=0
cudadec-memtype=0

{sink_section}
[osd]
enable=1
gpu-id=0
border-width=3
text-size=12
text-color=1;1;1;1;
text-bg-color=0.3;0.3;0.3;1
font=Serif
show-clock=0
nvbuf-memory-type=0

[streammux]
gpu-id=0
live-source={1 if live_source else 0}
batch-size=1
batched-push-timeout={DEEPSTREAM_STREAMMUX_BATCH_TIMEOUT}
width={DEEPSTREAM_STREAMMUX_WIDTH}
height={DEEPSTREAM_STREAMMUX_HEIGHT}
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file={infer_config_path}

[tests]
file-loop=0
"""
    with open(app_config_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("应用配置已生成: %s", app_config_path)
    return app_config_path


def _mode_str() -> str:
    """返回网络精度字符串"""
    return {0: "32", 1: "8", 2: "16"}.get(DEEPSTREAM_NETWORK_MODE, "32")


def generate_all_configs() -> dict:
    """为所有模型生成 DeepStream 配置文件

    Returns:
        dict: algCode -> {
            "labels_path": str,
            "infer_config_path": str,
            "onnx_path": str,
        }
    """
    os.makedirs(DEEPSTREAM_CONFIG_DIR, exist_ok=True)
    configs = {}

    for model_file, cfg in MODEL_CONFIGS.items():
        alg_code = cfg["algCode"]
        labels = cfg["labels"]
        num_classes = len(labels)

        # ONNX 文件路径（.pt 替换为 .onnx）
        onnx_file = model_file.rsplit(".", 1)[0] + ".onnx"
        onnx_path = os.path.join(MODEL_DIR, onnx_file)

        # 生成标签文件
        labels_path = generate_labels_file(
            alg_code, labels, DEEPSTREAM_CONFIG_DIR
        )

        # 生成推理配置
        infer_config_path = generate_infer_config(
            alg_code, onnx_path, labels_path, num_classes, DEEPSTREAM_CONFIG_DIR
        )

        configs[alg_code] = {
            "labels_path": labels_path,
            "infer_config_path": infer_config_path,
            "onnx_path": onnx_path,
        }

    logger.info("所有 DeepStream 配置文件已生成 (%d 个模型)", len(configs))
    return configs
