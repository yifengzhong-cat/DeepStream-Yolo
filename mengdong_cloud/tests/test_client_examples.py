#!/usr/bin/env python3
"""
蒙东云端AI分析平台 — API 客户端调用示例

本脚本展示如何使用 Python requests 库调用各 API 接口。
需要先启动服务，然后运行本脚本进行测试。

使用方式:
    # 1. 先启动服务（Docker 方式或本地方式）
    docker-compose up -d
    # 或者
    cd mengdong_cloud && python -m uvicorn app.main:app --host 0.0.0.0 --port 22266

    # 2. 运行测试
    python tests/test_client_examples.py                              # 运行全部示例
    python tests/test_client_examples.py --image                      # 仅图片分析
    python tests/test_client_examples.py --video                      # 仅视频文件
    python tests/test_client_examples.py --stream                     # 仅视频流
    python tests/test_client_examples.py --base-url http://gpu:22266  # 指定服务地址
    python tests/test_client_examples.py --image-path ./test.jpg      # 指定真实图片
"""

import argparse
import base64
import json
import sys
import time

import requests

# ============================================================
# 配置
# ============================================================

BASE_URL = "http://127.0.0.1:22266"
TEST_RTSP_URL = "rtsp://192.168.1.100:554/stream1"
TEST_VIDEO_FILE = "file:///app/videos/test.mp4"


def _print(title, resp):
    """格式化打印"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  HTTP {resp.status_code}")
    text = json.dumps(resp.json(), ensure_ascii=False, indent=2)
    if len(text) > 2000:
        text = text[:2000] + "\n  ...(truncated)"
    print(f"  {text}")


def _make_test_image():
    """生成 100x100 蓝色测试图片的 Base64"""
    try:
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (255, 0, 0)
        _, buf = cv2.imencode(".jpg", img)
        return base64.b64encode(buf).decode("utf-8")
    except ImportError:
        print("  [提示] 未安装 opencv-python，请指定 --image-path")
        sys.exit(1)


# ============================================================
# 示例函数
# ============================================================


def example_health():
    """
    GET /health

    cURL:
        curl http://127.0.0.1:22266/health
    """
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    _print("健康检查", r)


def example_abilities():
    """
    POST /v1/service/abilities

    cURL:
        curl -X POST http://127.0.0.1:22266/v1/service/abilities
    """
    r = requests.post(f"{BASE_URL}/v1/service/abilities", timeout=5)
    _print("算法能力查询", r)


def example_image(image_path=None):
    """
    POST /v1/service/imageTask — 图片分析

    cURL:
        IMG=$(base64 -w0 test.jpg)
        curl -X POST http://127.0.0.1:22266/v1/service/imageTask \\
             -H 'Content-Type: application/json' \\
             -d '{"analyseId":"img-001","algCode":"010101","imageData":"'$IMG'"}'
    """
    if image_path:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    else:
        img_b64 = _make_test_image()

    r = requests.post(f"{BASE_URL}/v1/service/imageTask", json={
        "analyseId": "img-example-001",
        "algCode": "010101",
        "imageData": img_b64,
    }, timeout=30)
    _print("图片分析（基本）", r)

    # 带灵敏度
    r2 = requests.post(f"{BASE_URL}/v1/service/imageTask", json={
        "analyseId": "img-example-002",
        "algCode": "010101",
        "imageData": img_b64,
        "rule": {"algParams": [{"key": "--sensitivity", "value": "4"}]},
    }, timeout=30)
    _print("图片分析（灵敏度=4）", r2)


def example_video_file():
    """
    POST /v1/service/videoTask — 视频文件分析

    cURL:
        curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
             -H 'Content-Type: application/json' \\
             -d '{
                "algCode":"010104","interval":60,"command":1,
                "videoInfo":[{
                    "analyseId":"video-001","devCode":"file",
                    "formatType":0,"videoUrl":"file:///app/videos/test.mp4"
                }]
             }'
    """
    r = requests.post(f"{BASE_URL}/v1/service/videoTask", json={
        "algCode": "010104",
        "interval": 60,
        "command": 1,
        "videoInfo": [{
            "analyseId": "video-file-001",
            "devCode": "file-input",
            "formatType": 0,
            "videoUrl": TEST_VIDEO_FILE,
        }],
    }, timeout=10)
    _print("视频文件分析（启动）", r)


def example_stream_lifecycle():
    """
    视频流分析完整生命周期：启动 → 停止 → 恢复 → 删除

    cURL (启动):
        curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
             -H 'Content-Type: application/json' \\
             -d '{
                "algCode":"010101","interval":30,"command":1,
                "videoInfo":[{
                    "analyseId":"stream-001","devCode":"cam-01",
                    "formatType":0,"videoUrl":"rtsp://192.168.1.100:554/stream1"
                }],
                "rule":{"algParams":[{"key":"--sensitivity","value":"3"}]}
             }'

    cURL (停止):
        curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
             -H 'Content-Type: application/json' \\
             -d '{"algCode":"010101","command":0,
                  "videoInfo":[{"analyseId":"stream-001","devCode":"cam-01","formatType":0,"videoUrl":""}]}'

    cURL (恢复):
        curl -X POST http://127.0.0.1:22266/v1/service/controlTask \\
             -H 'Content-Type: application/json' \\
             -d '{"analyseId":"stream-001","command":1}'

    cURL (删除):
        curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
             -H 'Content-Type: application/json' \\
             -d '{"algCode":"010101","command":2,
                  "videoInfo":[{"analyseId":"stream-001","devCode":"cam-01","formatType":0,"videoUrl":""}]}'
    """
    aid = f"stream-{int(time.time())}"

    # 1. 启动
    r = requests.post(f"{BASE_URL}/v1/service/videoTask", json={
        "algCode": "010101",
        "interval": 30,
        "command": 1,
        "videoInfo": [{
            "analyseId": aid,
            "devCode": "cam-01",
            "formatType": 0,
            "videoUrl": TEST_RTSP_URL,
        }],
        "rule": {"algParams": [{"key": "--sensitivity", "value": "3"}]},
    }, timeout=10)
    _print(f"[1/4] 启动视频流 {aid}", r)

    time.sleep(2)

    # 2. 停止
    r = requests.post(f"{BASE_URL}/v1/service/videoTask", json={
        "algCode": "010101",
        "command": 0,
        "videoInfo": [{"analyseId": aid, "devCode": "cam-01", "formatType": 0, "videoUrl": ""}],
    }, timeout=10)
    _print(f"[2/4] 停止视频流 {aid}", r)

    # 3. 恢复
    r = requests.post(f"{BASE_URL}/v1/service/controlTask", json={
        "analyseId": aid,
        "command": 1,
    }, timeout=10)
    _print(f"[3/4] 恢复任务 {aid}", r)

    # 4. 删除
    r = requests.post(f"{BASE_URL}/v1/service/videoTask", json={
        "algCode": "010101",
        "command": 2,
        "videoInfo": [{"analyseId": aid, "devCode": "cam-01", "formatType": 0, "videoUrl": ""}],
    }, timeout=10)
    _print(f"[4/4] 删除任务 {aid}", r)


# ============================================================
# 主入口
# ============================================================


def main():
    global BASE_URL, TEST_RTSP_URL

    p = argparse.ArgumentParser(description="蒙东AI平台 — API 调用示例")
    p.add_argument("--base-url", default=BASE_URL, help="服务地址")
    p.add_argument("--image", action="store_true", help="仅图片分析")
    p.add_argument("--video", action="store_true", help="仅视频文件")
    p.add_argument("--stream", action="store_true", help="仅视频流")
    p.add_argument("--image-path", help="图片文件路径")
    p.add_argument("--rtsp-url", help="RTSP 流地址")
    args = p.parse_args()

    BASE_URL = args.base_url
    if args.rtsp_url:
        TEST_RTSP_URL = args.rtsp_url

    run_all = not (args.image or args.video or args.stream)

    print(f"\n  服务地址: {BASE_URL}\n")

    try:
        if run_all:
            example_health()
            example_abilities()

        if run_all or args.image:
            example_image(args.image_path)

        if run_all or args.video:
            example_video_file()

        if run_all or args.stream:
            example_stream_lifecycle()

    except requests.ConnectionError:
        print(f"\n  ❌ 无法连接: {BASE_URL}")
        print("  请先启动服务: docker-compose up -d")
        sys.exit(1)

    print("\n  ✅ 完成\n")


if __name__ == "__main__":
    main()
