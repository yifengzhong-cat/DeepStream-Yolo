#!/bin/bash
# ============================================================
# 蒙东云端AI分析平台 - 手动启动脚本
#
# 用于在宿主机上直接启动服务（不使用 Docker），等效于 Dockerfile 中的：
#   CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "22266"]
#
# 前置条件：
#   1. 已安装 Python 3.8+ 及 pip
#   2. 已安装 pip 依赖:  pip install -r requirements.txt
#   3. 模型文件已放入 models/ 目录
#   4. （可选）已安装 NVIDIA DeepStream SDK + pyds（视频流 DeepStream 管道需要）
#   5. （可选）已编译 libnvdsinfer_custom_impl_Yolo.so（DeepStream 推理需要）
#
# 使用方式：
#   bash run.sh                        # 默认端口 22266
#   bash run.sh --port 8080            # 自定义端口
#   SERVICE_PORT=8080 bash run.sh      # 通过环境变量自定义
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认环境变量（如未设置则使用默认值） ----
export SERVICE_HOST="${SERVICE_HOST:-0.0.0.0}"
export SERVICE_PORT="${SERVICE_PORT:-22266}"
export MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
export DEVICE="${DEVICE:-0}"
export CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.5}"

# DeepStream 相关（仅在安装了 DeepStream SDK 时生效）
export DEEPSTREAM_YOLO_DIR="${DEEPSTREAM_YOLO_DIR:-${SCRIPT_DIR}/../}"
export DEEPSTREAM_CONFIG_DIR="${DEEPSTREAM_CONFIG_DIR:-${SCRIPT_DIR}/ds_configs}"
export DEEPSTREAM_NETWORK_MODE="${DEEPSTREAM_NETWORK_MODE:-0}"
export DEEPSTREAM_INFER_SIZE="${DEEPSTREAM_INFER_SIZE:-640}"

# 创建必要目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DEEPSTREAM_CONFIG_DIR"

# ---- 解析命令行参数 ----
HOST="$SERVICE_HOST"
PORT="$SERVICE_PORT"
WORKERS=1
RELOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            export SERVICE_HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            export SERVICE_PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --help|-h)
            echo "用法: bash run.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --host HOST     监听地址 (默认: 0.0.0.0)"
            echo "  --port PORT     监听端口 (默认: 22266)"
            echo "  --workers N     工作进程数 (默认: 1)"
            echo "  --reload        开发模式，代码变更自动重载"
            echo "  --help, -h      显示帮助信息"
            echo ""
            echo "环境变量:"
            echo "  SERVICE_HOST              监听地址 (默认: 0.0.0.0)"
            echo "  SERVICE_PORT              监听端口 (默认: 22266)"
            echo "  MODEL_DIR                 模型目录 (默认: ./models)"
            echo "  OUTPUT_DIR                输出目录 (默认: ./output)"
            echo "  DEVICE                    GPU设备号，'cpu'=CPU (默认: 0)"
            echo "  CONFIDENCE_THRESHOLD      置信度阈值 (默认: 0.5)"
            echo "  PLATFORM_HOST             上报平台地址 (默认: http://127.0.0.1:8080)"
            echo "  DEEPSTREAM_YOLO_DIR       DeepStream-Yolo 项目目录"
            echo "  DEEPSTREAM_CONFIG_DIR     DeepStream 配置文件目录"
            echo "  DEEPSTREAM_NETWORK_MODE   TensorRT精度: 0=FP32, 2=FP16 (默认: 0)"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ---- 启动信息 ----
echo "============================================"
echo "  蒙东云端AI分析平台 - 手动启动"
echo "============================================"
echo "  监听地址:   ${HOST}:${PORT}"
echo "  模型目录:   ${MODEL_DIR}"
echo "  输出目录:   ${OUTPUT_DIR}"
echo "  推理设备:   ${DEVICE}"
echo "  置信度阈值: ${CONFIDENCE_THRESHOLD}"
echo ""
echo "  Swagger文档: http://${HOST}:${PORT}/docs"
echo "  健康检查:    http://${HOST}:${PORT}/health"
echo "============================================"
echo ""

# ---- 启动 uvicorn ----
# 等效于 Dockerfile 中的:
#   CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "22266"]
exec python3 -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    $RELOAD
