# ============================================================
# 蒙东云端AI分析平台 Docker镜像
# 基于 NVIDIA PyTorch 镜像，支持 GPU 推理
# ============================================================

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="mengdong_cloud"
LABEL description="蒙东云端AI分析平台 - YOLOv8 推理服务"

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/

# 创建模型目录和输出目录
RUN mkdir -p /app/models /app/output

# 复制模型文件（构建时需要将模型放到 models/ 目录下）
COPY models/ /app/models/

# 环境变量
ENV SERVICE_HOST=0.0.0.0
ENV SERVICE_PORT=22266
ENV MODEL_DIR=/app/models
ENV OUTPUT_DIR=/app/output
ENV DEVICE=0
ENV CONFIDENCE_THRESHOLD=0.5

# 暴露端口
EXPOSE 22266

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:22266/health'); exit(0 if r.status_code==200 else 1)" || exit 1

# 启动服务
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "22266"]
