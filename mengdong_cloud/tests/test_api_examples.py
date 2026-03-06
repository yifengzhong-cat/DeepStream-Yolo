# ============================================================
# 蒙东云端AI分析平台 — API 接口测试示例
#
# 使用 pytest + httpx（FastAPI TestClient）对各接口进行测试。
# 这些测试可以在没有 GPU / DeepStream 的开发环境中运行，
# 通过 mock 模拟推理后端来验证 API 的请求/响应格式和业务逻辑。
#
# 运行方式:
#   cd mengdong_cloud
#   pip install pytest httpx
#   pytest tests/test_api_examples.py -v
# ============================================================

import base64
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 在导入 app 模块之前，设置环境变量以避免 /app 目录权限问题
_test_tmpdir = tempfile.mkdtemp(prefix="mengdong_test_")
os.environ.setdefault("MODEL_DIR", os.path.join(_test_tmpdir, "models"))
os.environ.setdefault("OUTPUT_DIR", os.path.join(_test_tmpdir, "output"))
os.environ.setdefault("DEEPSTREAM_CONFIG_DIR", os.path.join(_test_tmpdir, "ds_configs"))
os.environ.setdefault("DEEPSTREAM_YOLO_DIR", os.path.join(_test_tmpdir, "deepstream_yolo"))

from fastapi.testclient import TestClient

from app.inference import Detection


# ---- Fixtures ----


@pytest.fixture(autouse=True)
def mock_model_manager():
    """Mock model_manager 以跳过真实模型加载和 DeepStream 依赖"""
    with patch("app.main.model_manager") as mock_mm:
        mock_mm.is_loaded = True
        mock_mm.loaded_models = {"010101": {}, "010102": {}, "010103": {}, "010104": {}}
        mock_mm.ds_configs = {
            "010101": {"infer_config_path": "/tmp/fake.txt"},
            "010102": {"infer_config_path": "/tmp/fake.txt"},
            "010103": {"infer_config_path": "/tmp/fake.txt"},
            "010104": {"infer_config_path": "/tmp/fake.txt"},
        }
        mock_mm.load_all_models = MagicMock()
        yield mock_mm


@pytest.fixture
def client(mock_model_manager):
    """创建 FastAPI 测试客户端（跳过 lifespan 中的模型加载）"""
    from app.main import app

    # 使用 TestClient，lifespan 中的 model_manager.load_all_models 已被 mock
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def sample_image_base64():
    """生成一个简单的测试图片并返回 Base64 编码

    创建 100x100 蓝色图片用于接口测试。
    """
    import cv2

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # 蓝色 BGR
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


@pytest.fixture
def sample_detections():
    """模拟 DeepStream 推理返回的检测结果"""
    return [
        Detection(class_id=0, confidence=0.92, x1=10, y1=20, x2=50, y2=80),
        Detection(class_id=6, confidence=0.85, x1=15, y1=10, x2=45, y2=35),
    ]


# ============================================================
# 1. 健康检查 GET /health
# ============================================================


class TestHealthCheck:
    """健康检查接口测试"""

    def test_health_check(self, client):
        """测试 GET /health 返回服务状态"""
        resp = client.get("/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "loaded_count" in data
        assert "deepstream_configs" in data


# ============================================================
# 2. 算法能力查询 POST /v1/service/abilities
# ============================================================


class TestAbilities:
    """算法能力接口测试"""

    def test_get_abilities(self, client):
        """测试查询所有可用算法能力

        请求示例:
            POST /v1/service/abilities
            Body: {}
        """
        resp = client.post("/v1/service/abilities")
        assert resp.status_code == 200

        data = resp.json()
        assert data["resultCode"] == "200"
        assert "resultValue" in data

        ability_info = data["resultValue"]["abilityInfo"]
        # MODEL_CONFIGS 中定义了 4 个模型 (010101~010104)
        # 如果添加或删除模型，此处需同步更新
        assert ability_info["number"] == len(ability_info["ability"])
        assert ability_info["number"] > 0

        # 验证每个算法都有 algCode 和 algDesc
        for ability in ability_info["ability"]:
            assert "algCode" in ability
            assert "algDesc" in ability
            assert "algParams" in ability

    def test_abilities_include_sensitivity(self, client):
        """测试算法能力中包含灵敏度参数"""
        resp = client.post("/v1/service/abilities")
        data = resp.json()

        ability_list = data["resultValue"]["abilityInfo"]["ability"]
        for ability in ability_list:
            param_keys = [p["key"] for p in ability["algParams"]]
            assert "--sensitivity" in param_keys


# ============================================================
# 3. 图片分析 POST /v1/service/imageTask
# ============================================================


class TestImageTask:
    """图片分析接口测试

    图片分析流程：
    1. 客户端将图片 Base64 编码后发送
    2. 服务端使用 DeepStream 管道推理
    3. 返回检测结果、原始图片和标注图片 (Base64)
    """

    def test_image_task_success(self, client, sample_image_base64, sample_detections):
        """测试图片分析成功场景

        请求示例:
            POST /v1/service/imageTask
            Body:
            {
                "analyseId": "img-test-001",
                "algCode": "010101",
                "imageData": "<base64 encoded image>"
            }
        """
        with patch("app.routers.image_task.model_manager") as mock_mm:
            mock_mm.get_ds_config.return_value = {"infer_config_path": "/tmp/fake.txt"}
            mock_mm.predict.return_value = sample_detections

            resp = client.post(
                "/v1/service/imageTask",
                json={
                    "analyseId": "img-test-001",
                    "algCode": "010101",
                    "imageData": sample_image_base64,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"

        result = data["resultValue"]
        assert "analyseResults" in result
        assert "analyseTime" in result
        assert "rawImageData" in result
        assert "osdImageData" in result
        assert "resultDetail" in result

        # 验证检测到的标签
        assert len(result["analyseResults"]) > 0

        # 验证结果详情
        assert len(result["resultDetail"]) > 0
        for detail in result["resultDetail"]:
            assert "algCode" in detail
            assert "resultDesc" in detail
            assert "num" in detail
            assert "resultItems" in detail
            for item in detail["resultItems"]:
                assert "score" in item
                assert "leftTopX" in item
                assert "leftTopY" in item
                assert "rightBottomX" in item
                assert "rightBottomY" in item

    def test_image_task_with_sensitivity(
        self, client, sample_image_base64, sample_detections
    ):
        """测试图片分析带灵敏度参数

        请求示例（含灵敏度设置）:
            POST /v1/service/imageTask
            Body:
            {
                "analyseId": "img-test-002",
                "algCode": "010101",
                "imageData": "<base64>",
                "rule": {
                    "algParams": [
                        {"key": "--sensitivity", "value": "3"}
                    ]
                }
            }
        """
        with patch("app.routers.image_task.model_manager") as mock_mm:
            mock_mm.get_ds_config.return_value = {"infer_config_path": "/tmp/fake.txt"}
            mock_mm.predict.return_value = sample_detections

            resp = client.post(
                "/v1/service/imageTask",
                json={
                    "analyseId": "img-test-002",
                    "algCode": "010101",
                    "imageData": sample_image_base64,
                    "rule": {
                        "algParams": [{"key": "--sensitivity", "value": "3"}],
                    },
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"

    def test_image_task_invalid_image(self, client):
        """测试图片数据无效的情况

        请求示例（无效 Base64）:
            POST /v1/service/imageTask
            Body:
            {
                "analyseId": "img-test-err",
                "algCode": "010101",
                "imageData": "not-valid-base64!!"
            }
        """
        with patch("app.routers.image_task.model_manager") as mock_mm:
            mock_mm.get_ds_config.return_value = {"infer_config_path": "/tmp/fake.txt"}

            resp = client.post(
                "/v1/service/imageTask",
                json={
                    "analyseId": "img-test-err",
                    "algCode": "010101",
                    "imageData": "not-valid-base64!!",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "400"
        assert "解码失败" in data["resultHint"]

    def test_image_task_unknown_algcode(self, client, sample_image_base64):
        """测试使用不存在的算法编码

        请求示例（不存在的 algCode）:
            POST /v1/service/imageTask
            Body:
            {
                "analyseId": "img-test-404",
                "algCode": "999999",
                "imageData": "<base64>"
            }
        """
        with patch("app.routers.image_task.model_manager") as mock_mm:
            mock_mm.get_ds_config.return_value = None

            resp = client.post(
                "/v1/service/imageTask",
                json={
                    "analyseId": "img-test-404",
                    "algCode": "999999",
                    "imageData": sample_image_base64,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "404"


# ============================================================
# 4. 视频任务管理 POST /v1/service/videoTask
# ============================================================


class TestVideoTask:
    """视频任务管理接口测试

    支持三种操作:
    - command=1: 启动视频分析任务
    - command=0: 停止视频分析任务
    - command=2: 删除视频分析任务
    """

    def test_start_video_task(self, client):
        """测试启动视频/视频流分析任务

        请求示例（启动 RTSP 视频流分析）:
            POST /v1/service/videoTask
            Body:
            {
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
                ]
            }
        """
        with patch("app.routers.video_task.video_task_manager") as mock_vtm:
            mock_task = MagicMock()
            mock_task.osd_video_url = "/output/stream-001.mp4"
            mock_vtm.create_task.return_value = mock_task

            resp = client.post(
                "/v1/service/videoTask",
                json={
                    "algCode": "010101",
                    "interval": 30,
                    "command": 1,
                    "videoInfo": [
                        {
                            "analyseId": "stream-001",
                            "devCode": "camera-01",
                            "formatType": 0,
                            "videoUrl": "rtsp://192.168.1.100:554/stream1",
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert len(data["resultValue"]) == 1
        assert data["resultValue"][0]["analyseId"] == "stream-001"
        assert data["resultValue"][0]["devCode"] == "camera-01"
        assert "osdVideoUrl" in data["resultValue"][0]

    def test_start_video_file_task(self, client):
        """测试启动本地视频文件分析任务

        请求示例（分析本地视频文件）:
            POST /v1/service/videoTask
            Body:
            {
                "algCode": "010102",
                "interval": 60,
                "command": 1,
                "videoInfo": [
                    {
                        "analyseId": "video-file-001",
                        "devCode": "file-input",
                        "formatType": 0,
                        "videoUrl": "file:///app/videos/test_video.mp4"
                    }
                ]
            }
        """
        with patch("app.routers.video_task.video_task_manager") as mock_vtm:
            mock_task = MagicMock()
            mock_vtm.create_task.return_value = mock_task

            resp = client.post(
                "/v1/service/videoTask",
                json={
                    "algCode": "010102",
                    "interval": 60,
                    "command": 1,
                    "videoInfo": [
                        {
                            "analyseId": "video-file-001",
                            "devCode": "file-input",
                            "formatType": 0,
                            "videoUrl": "file:///app/videos/test_video.mp4",
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"][0]["analyseId"] == "video-file-001"

    def test_start_multiple_streams(self, client):
        """测试同时启动多路视频流分析

        请求示例（多路摄像头同时分析）:
            POST /v1/service/videoTask
            Body:
            {
                "algCode": "010101",
                "interval": 30,
                "command": 1,
                "videoInfo": [
                    {
                        "analyseId": "stream-A",
                        "devCode": "cam-A",
                        "formatType": 0,
                        "videoUrl": "rtsp://192.168.1.101/live"
                    },
                    {
                        "analyseId": "stream-B",
                        "devCode": "cam-B",
                        "formatType": 0,
                        "videoUrl": "rtsp://192.168.1.102/live"
                    }
                ]
            }
        """
        with patch("app.routers.video_task.video_task_manager") as mock_vtm:
            mock_task = MagicMock()
            mock_vtm.create_task.return_value = mock_task

            resp = client.post(
                "/v1/service/videoTask",
                json={
                    "algCode": "010101",
                    "interval": 30,
                    "command": 1,
                    "videoInfo": [
                        {
                            "analyseId": "stream-A",
                            "devCode": "cam-A",
                            "formatType": 0,
                            "videoUrl": "rtsp://192.168.1.101/live",
                        },
                        {
                            "analyseId": "stream-B",
                            "devCode": "cam-B",
                            "formatType": 0,
                            "videoUrl": "rtsp://192.168.1.102/live",
                        },
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert len(data["resultValue"]) == 2

    def test_start_video_with_rule(self, client):
        """测试启动视频流分析任务并设置灵敏度规则

        请求示例（含灵敏度和检测区域规则）:
            POST /v1/service/videoTask
            Body:
            {
                "algCode": "010101",
                "interval": 30,
                "command": 1,
                "videoInfo": [
                    {
                        "analyseId": "stream-rule-001",
                        "devCode": "cam-rule",
                        "formatType": 0,
                        "videoUrl": "rtsp://192.168.1.100/live"
                    }
                ],
                "rule": {
                    "algParams": [
                        {"key": "--sensitivity", "value": "4"}
                    ],
                    "ruleNum": 1,
                    "ruleProperty": [
                        {
                            "ruleId": 1,
                            "ruleDesc": "检测区域",
                            "pointNum": 4,
                            "point": [
                                {"id": 1, "x": 0, "y": 0},
                                {"id": 2, "x": 1920, "y": 0},
                                {"id": 3, "x": 1920, "y": 1080},
                                {"id": 4, "x": 0, "y": 1080}
                            ]
                        }
                    ]
                }
            }
        """
        with patch("app.routers.video_task.video_task_manager") as mock_vtm:
            mock_task = MagicMock()
            mock_vtm.create_task.return_value = mock_task

            resp = client.post(
                "/v1/service/videoTask",
                json={
                    "algCode": "010101",
                    "interval": 30,
                    "command": 1,
                    "videoInfo": [
                        {
                            "analyseId": "stream-rule-001",
                            "devCode": "cam-rule",
                            "formatType": 0,
                            "videoUrl": "rtsp://192.168.1.100/live",
                        }
                    ],
                    "rule": {
                        "algParams": [{"key": "--sensitivity", "value": "4"}],
                        "ruleNum": 1,
                        "ruleProperty": [
                            {
                                "ruleId": 1,
                                "ruleDesc": "检测区域",
                                "pointNum": 4,
                                "point": [
                                    {"id": 1, "x": 0, "y": 0},
                                    {"id": 2, "x": 1920, "y": 0},
                                    {"id": 3, "x": 1920, "y": 1080},
                                    {"id": 4, "x": 0, "y": 1080},
                                ],
                            }
                        ],
                    },
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"

    def test_stop_video_task(self, client):
        """测试停止视频分析任务

        请求示例:
            POST /v1/service/videoTask
            Body:
            {
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
            }
        """
        with patch("app.routers.video_task.video_task_manager") as mock_vtm:
            mock_vtm.stop_task.return_value = True

            resp = client.post(
                "/v1/service/videoTask",
                json={
                    "algCode": "010101",
                    "command": 0,
                    "videoInfo": [
                        {
                            "analyseId": "stream-001",
                            "devCode": "camera-01",
                            "formatType": 0,
                            "videoUrl": "",
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"][0]["analyseId"] == "stream-001"

    def test_delete_video_task(self, client):
        """测试删除视频分析任务

        请求示例:
            POST /v1/service/videoTask
            Body:
            {
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
            }
        """
        with patch("app.routers.video_task.video_task_manager") as mock_vtm:
            mock_vtm.delete_task.return_value = True

            resp = client.post(
                "/v1/service/videoTask",
                json={
                    "algCode": "010101",
                    "command": 2,
                    "videoInfo": [
                        {
                            "analyseId": "stream-001",
                            "devCode": "camera-01",
                            "formatType": 0,
                            "videoUrl": "",
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"


# ============================================================
# 5. 任务控制 POST /v1/service/controlTask
# ============================================================


class TestControlTask:
    """任务控制接口测试

    通过 command 字段控制视频分析任务:
    - command=0: 停止任务
    - command=1: 启动/恢复任务
    - command=2: 删除任务
    """

    def test_stop_task(self, client):
        """测试停止任务

        请求示例:
            POST /v1/service/controlTask
            Body: {"analyseId": "stream-001", "command": 0}
        """
        with patch("app.routers.control_task.video_task_manager") as mock_vtm:
            mock_vtm.stop_task.return_value = True

            resp = client.post(
                "/v1/service/controlTask",
                json={"analyseId": "stream-001", "command": 0},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultHint"] == "任务已停止"

    def test_resume_task(self, client):
        """测试恢复/启动任务

        请求示例:
            POST /v1/service/controlTask
            Body: {"analyseId": "stream-001", "command": 1}
        """
        with patch("app.routers.control_task.video_task_manager") as mock_vtm:
            mock_task = MagicMock()
            mock_task.is_running = False
            mock_vtm.get_task.return_value = mock_task

            resp = client.post(
                "/v1/service/controlTask",
                json={"analyseId": "stream-001", "command": 1},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultHint"] == "任务已启动"

    def test_delete_task(self, client):
        """测试删除任务

        请求示例:
            POST /v1/service/controlTask
            Body: {"analyseId": "stream-001", "command": 2}
        """
        with patch("app.routers.control_task.video_task_manager") as mock_vtm:
            mock_vtm.delete_task.return_value = True

            resp = client.post(
                "/v1/service/controlTask",
                json={"analyseId": "stream-001", "command": 2},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultHint"] == "任务已删除"

    def test_stop_nonexistent_task(self, client):
        """测试停止不存在的任务

        请求示例:
            POST /v1/service/controlTask
            Body: {"analyseId": "nonexistent", "command": 0}
        """
        with patch("app.routers.control_task.video_task_manager") as mock_vtm:
            mock_vtm.stop_task.return_value = False

            resp = client.post(
                "/v1/service/controlTask",
                json={"analyseId": "nonexistent", "command": 0},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "404"

    def test_invalid_command(self, client):
        """测试无效的控制指令

        请求示例:
            POST /v1/service/controlTask
            Body: {"analyseId": "stream-001", "command": 99}
        """
        resp = client.post(
            "/v1/service/controlTask",
            json={"analyseId": "stream-001", "command": 99},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "400"


# ============================================================
# 6. 保活心跳 POST /analysis/api/v1/keepAlive
# ============================================================


class TestKeepAlive:
    """保活心跳接口测试"""

    def test_keep_alive(self, client):
        """测试保活心跳

        请求示例:
            POST /analysis/api/v1/keepAlive
            Body: {"devIP": "192.168.1.50", "devPort": 22266}
        """
        resp = client.post(
            "/analysis/api/v1/keepAlive",
            json={"devIP": "192.168.1.50", "devPort": 22266},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"]["devId"] == "192.168.1.50:22266"


# ============================================================
# 7. 更新分析ID POST /analysis/api/v1/updateAnalyseID
# ============================================================


class TestUpdateAnalyseID:
    """更新分析ID接口测试"""

    def test_update_analyse_id_success(self, client):
        """测试更新分析ID成功

        请求示例:
            POST /analysis/api/v1/updateAnalyseID
            Body: {
                "oldAnalyseId": "stream-001",
                "newAnalyseId": "stream-001-v2"
            }
        """
        with patch("app.routers.update_analyse.video_task_manager") as mock_vtm:
            mock_vtm.update_analyse_id.return_value = True

            resp = client.post(
                "/analysis/api/v1/updateAnalyseID",
                json={
                    "oldAnalyseId": "stream-001",
                    "newAnalyseId": "stream-001-v2",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"]["newAnalyseId"] == "stream-001-v2"

    def test_update_analyse_id_not_found(self, client):
        """测试更新不存在的分析ID

        请求示例:
            POST /analysis/api/v1/updateAnalyseID
            Body: {
                "oldAnalyseId": "nonexistent",
                "newAnalyseId": "new-id"
            }
        """
        with patch("app.routers.update_analyse.video_task_manager") as mock_vtm:
            mock_vtm.update_analyse_id.return_value = False

            resp = client.post(
                "/analysis/api/v1/updateAnalyseID",
                json={
                    "oldAnalyseId": "nonexistent",
                    "newAnalyseId": "new-id",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "404"


# ============================================================
# 8. 样本上传 POST /v1/service/uploadSamples
# ============================================================


class TestUploadSamples:
    """样本上传接口测试"""

    def test_upload_sample(self, client):
        """测试样本上传

        请求示例:
            POST /v1/service/uploadSamples
            Body: {"fileId": "sample-001", "md5": "abc123def456"}
        """
        resp = client.post(
            "/v1/service/uploadSamples",
            json={"fileId": "sample-001", "md5": "abc123def456"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"]["fileId"] == "sample-001"
