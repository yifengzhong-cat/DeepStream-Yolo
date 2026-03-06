#!/bin/bash
# ============================================================
# 蒙东云端AI分析平台 - Docker 构建与部署脚本
#
# 构建流程：
# 1. 检查模型文件
# 2. 复制 DeepStream-Yolo 项目文件（nvdsinfer_custom_impl_Yolo, utils）
# 3. 构建 Docker 镜像（自动编译 libnvdsinfer_custom_impl_Yolo.so）
# 4. 导出为 mengdong_cloud.tar
# ============================================================

set -e

IMAGE_NAME="mengdong_cloud"
IMAGE_TAG="latest"
TAR_FILE="mengdong_cloud.tar"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  蒙东云端AI分析平台 - 构建部署脚本"
echo "  (集成 DeepStream-Yolo 推理管道)"
echo "============================================"

# [1/5] 检查模型文件
echo ""
echo "[1/5] 检查模型文件..."
MODELS_DIR="./models"
REQUIRED_MODELS=(
    "model_mengdong_raa_adjusted.pt"
    "model_mengdong_small_SRL.pt"
    "model_mengdong_tower.pt"
    "model_mengdong_scene_album.pt"
)

for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "${MODELS_DIR}/${model}" ]; then
        echo "  [警告] 模型文件不存在: ${MODELS_DIR}/${model}"
        echo "  请将模型文件放入 ${MODELS_DIR}/ 目录后重新构建"
    else
        echo "  [OK] ${model}"
    fi
done

# [2/5] 复制 DeepStream-Yolo 项目文件
echo ""
echo "[2/5] 复制 DeepStream-Yolo 项目文件..."
DS_YOLO_DIR="./deepstream_yolo"
mkdir -p "${DS_YOLO_DIR}"

# 复制 nvdsinfer_custom_impl_Yolo（自定义推理插件 — NvDsInferParseYolo）
if [ -d "${PROJECT_ROOT}/nvdsinfer_custom_impl_Yolo" ]; then
    cp -r "${PROJECT_ROOT}/nvdsinfer_custom_impl_Yolo" "${DS_YOLO_DIR}/"
    echo "  [OK] nvdsinfer_custom_impl_Yolo/"
else
    echo "  [错误] 未找到 ${PROJECT_ROOT}/nvdsinfer_custom_impl_Yolo"
    echo "  请确保在 DeepStream-Yolo 项目根目录下运行本脚本"
    exit 1
fi

# 复制 utils（export_yoloV8.py 等导出脚本）
if [ -d "${PROJECT_ROOT}/utils" ]; then
    cp -r "${PROJECT_ROOT}/utils" "${DS_YOLO_DIR}/"
    echo "  [OK] utils/"
else
    echo "  [错误] 未找到 ${PROJECT_ROOT}/utils"
    exit 1
fi

# [3/5] 构建 Docker 镜像
echo ""
echo "[3/5] 构建 Docker 镜像: ${IMAGE_NAME}:${IMAGE_TAG} ..."
echo "  (将自动编译 libnvdsinfer_custom_impl_Yolo.so)"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# [4/5] 导出为 tar 包
echo ""
echo "[4/5] 导出镜像为 ${TAR_FILE} ..."
docker save -o ${TAR_FILE} ${IMAGE_NAME}:${IMAGE_TAG}

# [5/5] 清理临时文件
echo ""
echo "[5/5] 清理临时文件..."
rm -rf "${DS_YOLO_DIR}"
echo "  [OK] 已清理 ${DS_YOLO_DIR}"

echo ""
echo "============================================"
echo "  构建完成！"
echo ""
echo "  部署说明："
echo "  1. 将 ${TAR_FILE} 拷贝到目标服务器"
echo "  2. 执行: docker load -i ${TAR_FILE}"
echo "  3. 执行: docker run -d --gpus all -p 22266:22266 --name mengdong_cloud ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  4. 访问 http://<服务器IP>:22266/docs 查看API文档"
echo "  5. 健康检查: curl http://<服务器IP>:22266/health"
echo ""
echo "  DeepStream-Yolo 集成说明："
echo "  - 服务启动时自动将 .pt 模型导出为 ONNX（使用 export_yoloV8.py）"
echo "  - 首次推理时 DeepStream 自动将 ONNX 转为 TensorRT engine"
echo "  - 视频流使用 DeepStream 管道加速推理"
echo "  - 图片推理使用 ultralytics 后端"
echo "============================================"

echo ""
echo "文件大小："
ls -lh ${TAR_FILE}
