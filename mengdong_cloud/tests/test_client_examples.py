#!/usr/bin/env python3
"""
蒙东云端AI分析平台 — API 客户端调用示例

本文件展示如何使用 Python requests 调用各接口。
可直接运行，也可作为 API 集成参考文档。

使用方式:
    # 修改 BASE_URL 为实际服务地址后运行
    python tests/test_client_examples.py

    # 仅运行图片分析示例
    python tests/test_client_examples.py --image

    # 仅运行视频流示例
    python tests/test_client_examples.py --video

    # 仅运行视频流管理示例（启动→控制→停止→删除）
    python tests/test_client_examples.py --stream
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

# 测试用 RTSP 流地址（替换为实际的摄像头地址）
TEST_RTSP_URL = "rtsp://192.168.1.100:554/stream1"

# 测试用视频文件路径（容器内路径）
TEST_VIDEO_FILE = "file:///app/videos/test.mp4"

# 测试用图片路径（本地路径，用于图片分析测试）
TEST_IMAGE_PATH = None  # 设置为 None 时自动生成测试图片


def _print_result(title: str, resp: requests.Response):
    """格式化打印请求结果"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  状态码: {resp.status_code}")
    data = resp.json()
    # 截断过长的 Base64 数据用于显示
    display_data = json.dumps(data, ensure_ascii=False, indent=2)
    if len(display_data) > 2000:
        display_data = display_data[:2000] + "\n  ... (已截断)"
    print(f"  响应: {display_data}")


def _make_test_image_base64() -> str:
    """生成测试图片 Base64（100x100 蓝色方块）"""
    try:
        import cv2
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (255, 0, 0)  # 蓝色 BGR
        _, buf = cv2.imencode(".jpg", img)
        return base64.b64encode(buf).decode("utf-8")
    except ImportError:
        # 无 OpenCV 时使用一个最小 JPEG 的 Base64
        print("  [提示] 未安装 opencv-python，使用最小 JPEG 替代")
        # 1x1 白色 JPEG 最小有效数据
        return base64.b64encode(
            bytes.fromhex(
                "ffd8ffe000104a46494600010100000100010000"
                "ffdb004300080606070605080707070909080a0c14"
                "0d0c0b0b0c1912130f141d1a1f1e1d1a1c1c2024"
                "2e2720222c231c1c2837292c30313434341f27393d"
                "38323c2e333432ffc0000b080001000101011100"
                "ffc4001f00000105010101010101000000000000"
                "00000102030405060708090a0bffc400b5100002"
                "01030302040305050404000001770001020311"
                "04052131061241510761711322328108144291"
                "a1b1c109233352f0156272d10a162434e125f1"
                "1718191a262728292a35363738393a434445464"
                "748494a535455565758595a636465666768696a"
                "737475767778797a838485868788898a929394"
                "95969798999aa2a3a4a5a6a7a8a9aab2b3b4b5"
                "b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5"
                "d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3"
                "f4f5f6f7f8f9faffda0008010100003f00fbdd"
                "69a28a0028a2800affd9"
            )
        ).decode("utf-8")


# ============================================================
# 示例 1: 健康检查
# ============================================================


def example_health_check():
    """
    GET /health — 检查服务状态

    cURL 等效命令:
        curl http://127.0.0.1:22266/health
    """
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    _print_result("健康检查 GET /health", resp)
    return resp.json()


# ============================================================
# 示例 2: 查询算法能力
# ============================================================


def example_get_abilities():
    """
    POST /v1/service/abilities — 查询服务支持的所有算法

    cURL 等效命令:
        curl -X POST http://127.0.0.1:22266/v1/service/abilities \\
             -H "Content-Type: application/json"

    响应示例:
    {
        "resultCode": "200",
        "resultValue": {
            "abilityInfo": {
                "number": 4,
                "ability": [
                    {
                        "algCode": "010101",
                        "algDesc": "人员穿戴检测（安全帽、安全带）",
                        "algParams": [
                            {"key": "--sensitivity", "value": "灵敏度,范围[1,5]"}
                        ]
                    },
                    ...
                ]
            }
        }
    }
    """
    resp = requests.post(f"{BASE_URL}/v1/service/abilities", timeout=5)
    _print_result("算法能力查询 POST /v1/service/abilities", resp)
    return resp.json()


# ============================================================
# 示例 3: 图片分析
# ============================================================


def example_image_analysis(image_path: str = None):
    """
    POST /v1/service/imageTask — 对单张图片进行 AI 推理

    cURL 等效命令:
        # 先将图片 Base64 编码
        IMG_B64=$(base64 -w0 test.jpg)

        curl -X POST http://127.0.0.1:22266/v1/service/imageTask \\
             -H "Content-Type: application/json" \\
             -d '{
                "analyseId": "img-001",
                "algCode": "010101",
                "imageData": "'$IMG_B64'"
             }'

    请求参数:
        analyseId   - 唯一分析ID
        algCode     - 算法编码 (010101: 人员穿戴, 010102: 钩子差速器,
                      010103: 铁塔, 010104: 场景设备)
        imageData   - Base64 编码的 JPEG/PNG 图片
        rule        - 可选，设置灵敏度等参数

    响应说明:
        resultValue.analyseResults   - 检测到的目标类别列表
        resultValue.rawImageData     - 原始图片 Base64
        resultValue.osdImageData     - 标注后图片 Base64
        resultValue.resultDetail     - 每个类别的检测结果详情
          └── resultItems            - 每个检测框的位置和置信度
    """
    # 准备图片
    if image_path:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    else:
        img_b64 = _make_test_image_base64()

    # --- 3a: 基本图片分析 ---
    payload = {
        "analyseId": "img-example-001",
        "algCode": "010101",  # 人员穿戴检测
        "imageData": img_b64,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/service/imageTask", json=payload, timeout=30
    )
    _print_result("图片分析（基本） POST /v1/service/imageTask", resp)

    # --- 3b: 带灵敏度参数的图片分析 ---
    payload_with_rule = {
        "analyseId": "img-example-002",
        "algCode": "010101",
        "imageData": img_b64,
        "rule": {
            "algParams": [
                {"key": "--sensitivity", "value": "4"}  # 灵敏度 4（较高）
            ],
        },
    }
    resp2 = requests.post(
        f"{BASE_URL}/v1/service/imageTask", json=payload_with_rule, timeout=30
    )
    _print_result("图片分析（含灵敏度） POST /v1/service/imageTask", resp2)

    return resp.json()


# ============================================================
# 示例 4: 视频/视频流分析（完整生命周期）
# ============================================================


def example_video_stream_lifecycle():
    """
    视频流分析完整生命周期：启动 → 查询 → 停止 → 删除

    支持的视频源:
    - RTSP 流: rtsp://ip:port/path
    - HTTP 流: http://ip:port/path
    - 本地文件: file:///path/to/video.mp4
    """
    analyse_id = f"stream-example-{int(time.time())}"

    # --- 4a: 启动 RTSP 视频流分析 ---
    print("\n" + "=" * 60)
    print("  步骤 1/4: 启动视频流分析任务")
    print("=" * 60)
    start_payload = {
        "algCode": "010101",  # 人员穿戴检测
        "interval": 30,  # 每30秒上报一次结果
        "command": 1,  # 1=启动
        "videoInfo": [
            {
                "analyseId": analyse_id,
                "devCode": "camera-example-01",
                "formatType": 0,
                "videoUrl": TEST_RTSP_URL,
            }
        ],
        "rule": {
            "algParams": [
                {"key": "--sensitivity", "value": "3"}
            ],
        },
    }
    """
    cURL 等效命令:
        curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
             -H "Content-Type: application/json" \\
             -d '{
                "algCode": "010101",
                "interval": 30,
                "command": 1,
                "videoInfo": [
                    {
                        "analyseId": "stream-001",
                        "devCode": "camera-01",
                        "formatType": 0,
                        "videoUrl": "rtsp://192.168.1.100:554/stream1"
                    }
                ],
                "rule": {
                    "algParams": [{"key": "--sensitivity", "value": "3"}]
                }
             }'
    """
    resp = requests.post(
        f"{BASE_URL}/v1/service/videoTask", json=start_payload, timeout=10
    )
    _print_result(f"启动视频流: {analyse_id}", resp)

    # --- 4b: 等待并查看健康状态 ---
    print("\n  等待 3 秒后检查服务状态...")
    time.sleep(3)
    health = requests.get(f"{BASE_URL}/health", timeout=5)
    _print_result("服务状态检查", health)

    # --- 4c: 停止视频流分析 ---
    print("\n" + "=" * 60)
    print("  步骤 2/4: 停止视频流分析任务")
    print("=" * 60)
    stop_payload = {
        "algCode": "010101",
        "command": 0,  # 0=停止
        "videoInfo": [
            {
                "analyseId": analyse_id,
                "devCode": "camera-example-01",
                "formatType": 0,
                "videoUrl": "",
            }
        ],
    }
    """
    cURL 等效命令:
        curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
             -H "Content-Type: application/json" \\
             -d '{
                "algCode": "010101",
                "command": 0,
                "videoInfo": [
                    {
                        "analyseId": "stream-001",
                        "devCode": "camera-01",
                        "formatType": 0,
                        "videoUrl": ""
                    }
                ]
             }'
    """
    resp = requests.post(
        f"{BASE_URL}/v1/service/videoTask", json=stop_payload, timeout=10
    )
    _print_result(f"停止视频流: {analyse_id}", resp)

    # --- 4d: 通过任务控制接口恢复任务 ---
    print("\n" + "=" * 60)
    print("  步骤 3/4: 通过控制接口管理任务")
    print("=" * 60)
    """
    cURL 等效命令（恢复任务）:
        curl -X POST http://127.0.0.1:22266/v1/service/controlTask \\
             -H "Content-Type: application/json" \\
             -d '{"analyseId": "stream-001", "command": 1}'
    """
    control_payload = {"analyseId": analyse_id, "command": 1}
    resp = requests.post(
        f"{BASE_URL}/v1/service/controlTask", json=control_payload, timeout=10
    )
    _print_result(f"恢复任务: {analyse_id}", resp)

    # --- 4e: 删除视频流分析任务 ---
    print("\n" + "=" * 60)
    print("  步骤 4/4: 删除视频流分析任务")
    print("=" * 60)
    delete_payload = {
        "algCode": "010101",
        "command": 2,  # 2=删除
        "videoInfo": [
            {
                "analyseId": analyse_id,
                "devCode": "camera-example-01",
                "formatType": 0,
                "videoUrl": "",
            }
        ],
    }
    """
    cURL 等效命令:
        curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
             -H "Content-Type: application/json" \\
             -d '{
                "algCode": "010101",
                "command": 2,
                "videoInfo": [
                    {
                        "analyseId": "stream-001",
                        "devCode": "camera-01",
                        "formatType": 0,
                        "videoUrl": ""
                    }
                ]
             }'
    """
    resp = requests.post(
        f"{BASE_URL}/v1/service/videoTask", json=delete_payload, timeout=10
    )
    _print_result(f"删除任务: {analyse_id}", resp)


# ============================================================
# 示例 5: 视频文件分析
# ============================================================


def example_video_file_analysis():
    """
    对本地视频文件进行分析

    cURL 等效命令:
        curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
             -H "Content-Type: application/json" \\
             -d '{
                "algCode": "010104",
                "interval": 60,
                "command": 1,
                "videoInfo": [
                    {
                        "analyseId": "video-file-001",
                        "devCode": "file-input",
                        "formatType": 0,
                        "videoUrl": "file:///app/videos/test.mp4"
                    }
                ]
             }'
    """
    payload = {
        "algCode": "010104",  # 场景设备检测
        "interval": 60,
        "command": 1,
        "videoInfo": [
            {
                "analyseId": "video-file-example-001",
                "devCode": "file-input",
                "formatType": 0,
                "videoUrl": TEST_VIDEO_FILE,
            }
        ],
    }
    resp = requests.post(
        f"{BASE_URL}/v1/service/videoTask", json=payload, timeout=10
    )
    _print_result("视频文件分析 POST /v1/service/videoTask", resp)
    return resp.json()


# ============================================================
# 示例 6: 保活心跳
# ============================================================


def example_keep_alive():
    """
    POST /analysis/api/v1/keepAlive — 服务保活心跳

    cURL 等效命令:
        curl -X POST http://127.0.0.1:22266/analysis/api/v1/keepAlive \\
             -H "Content-Type: application/json" \\
             -d '{"devIP": "192.168.1.50", "devPort": 22266}'
    """
    payload = {"devIP": "192.168.1.50", "devPort": 22266}
    resp = requests.post(
        f"{BASE_URL}/analysis/api/v1/keepAlive", json=payload, timeout=5
    )
    _print_result("保活心跳 POST /analysis/api/v1/keepAlive", resp)
    return resp.json()


# ============================================================
# 示例 7: 更新分析ID
# ============================================================


def example_update_analyse_id():
    """
    POST /analysis/api/v1/updateAnalyseID — 更新分析ID

    cURL 等效命令:
        curl -X POST http://127.0.0.1:22266/analysis/api/v1/updateAnalyseID \\
             -H "Content-Type: application/json" \\
             -d '{
                "oldAnalyseId": "stream-001",
                "newAnalyseId": "stream-001-v2"
             }'
    """
    payload = {
        "oldAnalyseId": "stream-old-001",
        "newAnalyseId": "stream-new-001",
    }
    resp = requests.post(
        f"{BASE_URL}/analysis/api/v1/updateAnalyseID", json=payload, timeout=5
    )
    _print_result("更新分析ID POST /analysis/api/v1/updateAnalyseID", resp)
    return resp.json()


# ============================================================
# 主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="蒙东云端AI分析平台 — API 接口调用示例"
    )
    parser.add_argument(
        "--base-url", default=BASE_URL, help=f"服务地址 (默认: {BASE_URL})"
    )
    parser.add_argument("--image", action="store_true", help="仅运行图片分析示例")
    parser.add_argument("--video", action="store_true", help="仅运行视频文件分析示例")
    parser.add_argument("--stream", action="store_true", help="仅运行视频流生命周期示例")
    parser.add_argument(
        "--image-path", default=None, help="用于图片分析的图片文件路径"
    )
    parser.add_argument(
        "--rtsp-url", default=None, help="用于视频流测试的 RTSP 地址"
    )

    args = parser.parse_args()

    global BASE_URL, TEST_RTSP_URL
    BASE_URL = args.base_url
    if args.rtsp_url:
        TEST_RTSP_URL = args.rtsp_url

    run_all = not (args.image or args.video or args.stream)

    print(f"\n{'#'*60}")
    print(f"  蒙东云端AI分析平台 — API 接口调用示例")
    print(f"  服务地址: {BASE_URL}")
    print(f"{'#'*60}")

    try:
        if run_all:
            example_health_check()
            example_get_abilities()
            example_keep_alive()

        if run_all or args.image:
            example_image_analysis(args.image_path)

        if run_all or args.video:
            example_video_file_analysis()

        if run_all or args.stream:
            example_video_stream_lifecycle()

        if run_all:
            example_update_analyse_id()

    except requests.ConnectionError:
        print(f"\n  ❌ 无法连接到服务: {BASE_URL}")
        print("  请确保服务已启动。")
        print("  启动方式: cd mengdong_cloud && python -m uvicorn app.main:app --port 22266")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print("  所有示例执行完成")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
