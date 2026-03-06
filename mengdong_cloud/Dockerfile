# ============================================================
# 蒙东云端AI分析平台 Docker镜像
# 基于 NVIDIA DeepStream 镜像，使用 DeepStream-Yolo 推理管道
# 所有推理（视频流 + 图片）均通过 DeepStream 管道完成
# ============================================================

FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

LABEL maintainer="mengdong_cloud"
LABEL description="蒙东云端AI分析平台 - DeepStream-Yolo + YOLOv8 推理服务"

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖和 Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-gi \
    gir1.2-gst-rtsp-server-1.0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ============================================================
# 复制 DeepStream-Yolo 项目文件并编译自定义推理插件
# ============================================================

# 复制 DeepStream-Yolo 的自定义推理插件源码和工具
COPY deepstream_yolo/nvdsinfer_custom_impl_Yolo/ /app/deepstream_yolo/nvdsinfer_custom_impl_Yolo/
COPY deepstream_yolo/utils/ /app/deepstream_yolo/utils/

# 编译自定义推理库 libnvdsinfer_custom_impl_Yolo.so
# （参照 DeepStream-Yolo 项目的 Makefile）
RUN cd /app/deepstream_yolo && \
    CUDA_VER=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+') && \
    echo "编译 DeepStream-Yolo 自定义推理插件 (CUDA ${CUDA_VER})..." && \
    make -C nvdsinfer_custom_impl_Yolo CUDA_VER=${CUDA_VER} && \
    echo "编译完成: libnvdsinfer_custom_impl_Yolo.so"

# ============================================================
# 复制应用代码和模型
# ============================================================

COPY app/ ./app/

# 创建目录
RUN mkdir -p /app/models /app/output /app/ds_configs

# 复制模型文件（构建时需要将模型放到 models/ 目录下）
COPY models/ /app/models/

# ============================================================
# 环境变量
# ============================================================
ENV SERVICE_HOST=0.0.0.0
ENV SERVICE_PORT=22266
ENV MODEL_DIR=/app/models
ENV OUTPUT_DIR=/app/output
ENV DEVICE=0
ENV CONFIDENCE_THRESHOLD=0.5
ENV DEEPSTREAM_YOLO_DIR=/app/deepstream_yolo
ENV DEEPSTREAM_CONFIG_DIR=/app/ds_configs
ENV DEEPSTREAM_NETWORK_MODE=0
ENV DEEPSTREAM_INFER_SIZE=640

# 暴露端口
EXPOSE 22266

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import requests; r=requests.get('http://localhost:22266/health'); exit(0 if r.status_code==200 else 1)" || exit 1

# 启动服务
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "22266"]
