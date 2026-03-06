#!/bin/bash
# ============================================================
# 蒙东云端AI分析平台 - Docker 构建与部署脚本
# ============================================================

set -e

IMAGE_NAME="mengdong_cloud"
IMAGE_TAG="latest"
TAR_FILE="mengdong_cloud.tar"

echo "============================================"
echo "  蒙东云端AI分析平台 - 构建部署脚本"
echo "============================================"

# 检查模型文件
echo ""
echo "[1/4] 检查模型文件..."
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

# 构建 Docker 镜像
echo ""
echo "[2/4] 构建 Docker 镜像: ${IMAGE_NAME}:${IMAGE_TAG} ..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# 导出为 tar 包
echo ""
echo "[3/4] 导出镜像为 ${TAR_FILE} ..."
docker save -o ${TAR_FILE} ${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "[4/4] 构建完成！"
echo ""
echo "============================================"
echo "  部署说明："
echo "  1. 将 ${TAR_FILE} 拷贝到目标服务器"
echo "  2. 执行: docker load -i ${TAR_FILE}"
echo "  3. 执行: docker run -d --gpus all -p 22266:22266 --name mengdong_cloud ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  4. 访问 http://<服务器IP>:22266/docs 查看API文档"
echo "  5. 健康检查: curl http://<服务器IP>:22266/health"
echo "============================================"

echo ""
echo "文件大小："
ls -lh ${TAR_FILE}
