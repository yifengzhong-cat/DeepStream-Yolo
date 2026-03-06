#!/usr/bin/env python3
# ============================================================
# 蒙东云端AI分析平台 — 图片/视频/视频流 接口测试示例
#
# 运行方式:
#   cd mengdong_cloud
#   pip install pytest httpx opencv-python-headless numpy
#   pytest tests/test_api_examples.py -v
#
# 测试说明:
#   - 所有测试通过 mock 模拟 DeepStream 推理后端，无需 GPU
#   - 每个测试方法的 docstring 中包含请求/响应格式说明和 cURL 示例
#   - 可在 Docker 容器或本地开发环境中运行
# ============================================================

from unittest.mock import MagicMock

from app.inference import Detection


# ============================================================
# 1. 健康检查
# ============================================================


class TestHealthCheck:
    """健康检查接口"""

    def test_health_check(self, client):
        """GET /health — 检查服务是否正常运行

        cURL:
            curl http://127.0.0.1:22266/health

        期望响应:
            {"status": "ok", "models_loaded": true, "loaded_count": 4, "deepstream_configs": 4}
        """
        resp = client.get("/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "loaded_count" in data
        assert "deepstream_configs" in data


# ============================================================
# 2. 算法能力
# ============================================================


class TestAbilities:
    """算法能力查询接口"""

    def test_get_abilities(self, client):
        """POST /v1/service/abilities — 查询所有可用算法

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/abilities

        期望响应:
            {
                "resultCode": "200",
                "resultValue": {
                    "abilityInfo": {
                        "number": 4,
                        "ability": [
                            {"algCode": "010101", "algDesc": "人员穿戴检测（安全帽、安全带）", ...},
                            ...
                        ]
                    }
                }
            }
        """
        resp = client.post("/v1/service/abilities")
        assert resp.status_code == 200

        data = resp.json()
        assert data["resultCode"] == "200"

        ability_info = data["resultValue"]["abilityInfo"]
        assert ability_info["number"] == len(ability_info["ability"])
        assert ability_info["number"] > 0

        for ability in ability_info["ability"]:
            assert "algCode" in ability
            assert "algDesc" in ability
            param_keys = [p["key"] for p in ability["algParams"]]
            assert "--sensitivity" in param_keys


# ============================================================
# 3. 图片分析接口 ★
# ============================================================


class TestImageTask:
    """图片分析接口测试

    接口：POST /v1/service/imageTask
    功能：对单张 Base64 编码的图片执行 AI 推理分析

    请求参数:
        analyseId  - 分析任务唯一ID
        algCode    - 算法编码（010101=人员穿戴, 010102=钩子差速器, 010103=铁塔, 010104=场景设备）
        imageData  - Base64 编码的 JPEG/PNG 图片
        rule       - 可选，灵敏度设置等规则
    """

    def test_image_basic(self, client, sample_image_base64, sample_detections):
        """基本图片分析

        cURL:
            IMG_B64=$(base64 -w0 test.jpg)
            curl -X POST http://127.0.0.1:22266/v1/service/imageTask \\
                 -H "Content-Type: application/json" \\
                 -d '{
                    "analyseId": "img-001",
                    "algCode": "010101",
                    "imageData": "'$IMG_B64'"
                 }'

        期望响应:
            {
                "resultCode": "200",
                "resultValue": {
                    "analyseResults": ["人", "黄色安全帽"],
                    "analyseTime": "2025-01-01 12:00:00",
                    "rawImageData": "<base64>",
                    "osdImageData": "<base64>",
                    "resultDetail": [
                        {
                            "algCode": "010101",
                            "resultDesc": "人",
                            "num": 1,
                            "resultItems": [
                                {"score": 92.0, "leftTopX": 10, "leftTopY": 20,
                                 "rightBottomX": 50, "rightBottomY": 80}
                            ]
                        }
                    ]
                }
            }
        """
        # 设置 mock 返回检测结果
        client._mock_mm.predict.return_value = sample_detections

        resp = client.post(
            "/v1/service/imageTask",
            json={
                "analyseId": "img-001",
                "algCode": "010101",
                "imageData": sample_image_base64,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"

        result = data["resultValue"]
        assert len(result["analyseResults"]) > 0
        assert result["rawImageData"] != ""
        assert result["osdImageData"] != ""
        assert result["analyseTime"] != ""

        # 验证结果详情结构
        for detail in result["resultDetail"]:
            assert detail["algCode"] == "010101"
            assert detail["num"] > 0
            for item in detail["resultItems"]:
                assert "score" in item
                assert "leftTopX" in item
                assert "leftTopY" in item
                assert "rightBottomX" in item
                assert "rightBottomY" in item

    def test_image_with_sensitivity(self, client, sample_image_base64, sample_detections):
        """图片分析 + 灵敏度参数

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/imageTask \\
                 -H "Content-Type: application/json" \\
                 -d '{
                    "analyseId": "img-002",
                    "algCode": "010101",
                    "imageData": "'$IMG_B64'",
                    "rule": {
                        "algParams": [{"key": "--sensitivity", "value": "4"}]
                    }
                 }'
        """
        client._mock_mm.predict.return_value = sample_detections

        resp = client.post(
            "/v1/service/imageTask",
            json={
                "analyseId": "img-002",
                "algCode": "010101",
                "imageData": sample_image_base64,
                "rule": {
                    "algParams": [{"key": "--sensitivity", "value": "4"}],
                },
            },
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "200"

    def test_image_no_detections(self, client, sample_image_base64):
        """图片分析无检测结果（空场景）"""
        client._mock_mm.predict.return_value = []

        resp = client.post(
            "/v1/service/imageTask",
            json={
                "analyseId": "img-003",
                "algCode": "010101",
                "imageData": sample_image_base64,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"]["analyseResults"] == []
        assert data["resultValue"]["resultDetail"] == []

    def test_image_invalid_base64(self, client):
        """图片 Base64 数据无效时返回 400

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/imageTask \\
                 -H "Content-Type: application/json" \\
                 -d '{"analyseId": "err-001", "algCode": "010101", "imageData": "not-valid!!"}'
        """
        resp = client.post(
            "/v1/service/imageTask",
            json={
                "analyseId": "err-001",
                "algCode": "010101",
                "imageData": "not-valid-base64!!",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "400"
        assert "解码失败" in data["resultHint"]

    def test_image_unknown_algcode(self, client, sample_image_base64):
        """使用不存在的算法编码返回 404

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/imageTask \\
                 -H "Content-Type: application/json" \\
                 -d '{"analyseId": "err-002", "algCode": "999999", "imageData": "..."}'
        """
        client._mock_mm.get_ds_config.return_value = None

        resp = client.post(
            "/v1/service/imageTask",
            json={
                "analyseId": "err-002",
                "algCode": "999999",
                "imageData": sample_image_base64,
            },
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "404"

    def test_image_inference_error(self, client, sample_image_base64):
        """推理异常时返回 500"""
        client._mock_mm.predict.side_effect = RuntimeError("DeepStream 推理失败")

        resp = client.post(
            "/v1/service/imageTask",
            json={
                "analyseId": "err-003",
                "algCode": "010101",
                "imageData": sample_image_base64,
            },
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "500"


# ============================================================
# 4. 视频任务接口 ★
# ============================================================


class TestVideoTask:
    """视频/视频流 任务管理接口

    接口：POST /v1/service/videoTask
    功能：管理视频分析任务的生命周期

    command 值:
        1 = 启动分析任务
        0 = 停止分析任务
        2 = 删除分析任务

    支持的视频源:
        - RTSP 流: rtsp://ip:port/path
        - HTTP 流: http://ip:port/path.m3u8
        - 本地文件: file:///path/to/video.mp4
    """

    def test_start_rtsp_stream(self, client):
        """启动 RTSP 视频流分析

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
                 -H "Content-Type: application/json" \\
                 -d '{
                    "algCode": "010101",
                    "interval": 30,
                    "command": 1,
                    "videoInfo": [{
                        "analyseId": "stream-001",
                        "devCode": "camera-01",
                        "formatType": 0,
                        "videoUrl": "rtsp://192.168.1.100:554/stream1"
                    }]
                 }'

        期望响应:
            {
                "resultCode": "200",
                "resultValue": [{
                    "analyseId": "stream-001",
                    "devCode": "camera-01",
                    "osdVideoUrl": "http://127.0.0.1:22266/output/stream-001.mp4"
                }]
            }
        """
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

    def test_start_video_file(self, client):
        """启动本地视频文件分析

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
                 -H "Content-Type: application/json" \\
                 -d '{
                    "algCode": "010102",
                    "interval": 60,
                    "command": 1,
                    "videoInfo": [{
                        "analyseId": "video-001",
                        "devCode": "file-input",
                        "formatType": 0,
                        "videoUrl": "file:///app/videos/test.mp4"
                    }]
                 }'
        """
        resp = client.post(
            "/v1/service/videoTask",
            json={
                "algCode": "010102",
                "interval": 60,
                "command": 1,
                "videoInfo": [
                    {
                        "analyseId": "video-001",
                        "devCode": "file-input",
                        "formatType": 0,
                        "videoUrl": "file:///app/videos/test.mp4",
                    }
                ],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"][0]["analyseId"] == "video-001"

    def test_start_multiple_streams(self, client):
        """同时启动多路视频流分析

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
                 -H "Content-Type: application/json" \\
                 -d '{
                    "algCode": "010101",
                    "interval": 30,
                    "command": 1,
                    "videoInfo": [
                        {"analyseId": "cam-A", "devCode": "A", "formatType": 0,
                         "videoUrl": "rtsp://192.168.1.101/live"},
                        {"analyseId": "cam-B", "devCode": "B", "formatType": 0,
                         "videoUrl": "rtsp://192.168.1.102/live"}
                    ]
                 }'
        """
        resp = client.post(
            "/v1/service/videoTask",
            json={
                "algCode": "010101",
                "interval": 30,
                "command": 1,
                "videoInfo": [
                    {
                        "analyseId": "cam-A",
                        "devCode": "A",
                        "formatType": 0,
                        "videoUrl": "rtsp://192.168.1.101/live",
                    },
                    {
                        "analyseId": "cam-B",
                        "devCode": "B",
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

    def test_start_stream_with_rule(self, client):
        """启动视频流分析（含灵敏度和检测区域规则）

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
                 -H "Content-Type: application/json" \\
                 -d '{
                    "algCode": "010101",
                    "interval": 30,
                    "command": 1,
                    "videoInfo": [{
                        "analyseId": "stream-rule",
                        "devCode": "cam-rule",
                        "formatType": 0,
                        "videoUrl": "rtsp://192.168.1.100/live"
                    }],
                    "rule": {
                        "algParams": [{"key": "--sensitivity", "value": "4"}],
                        "ruleNum": 1,
                        "ruleProperty": [{
                            "ruleId": 1, "ruleDesc": "检测区域", "pointNum": 4,
                            "point": [
                                {"id": 1, "x": 0, "y": 0},
                                {"id": 2, "x": 1920, "y": 0},
                                {"id": 3, "x": 1920, "y": 1080},
                                {"id": 4, "x": 0, "y": 1080}
                            ]
                        }]
                    }
                 }'
        """
        resp = client.post(
            "/v1/service/videoTask",
            json={
                "algCode": "010101",
                "interval": 30,
                "command": 1,
                "videoInfo": [
                    {
                        "analyseId": "stream-rule",
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
        assert resp.json()["resultCode"] == "200"

    def test_stop_video_task(self, client):
        """停止视频分析任务

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
                 -H "Content-Type: application/json" \\
                 -d '{
                    "algCode": "010101",
                    "command": 0,
                    "videoInfo": [{
                        "analyseId": "stream-001",
                        "devCode": "camera-01",
                        "formatType": 0,
                        "videoUrl": ""
                    }]
                 }'
        """
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
        """删除视频分析任务

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/videoTask \\
                 -H "Content-Type: application/json" \\
                 -d '{
                    "algCode": "010101",
                    "command": 2,
                    "videoInfo": [{
                        "analyseId": "stream-001",
                        "devCode": "camera-01",
                        "formatType": 0,
                        "videoUrl": ""
                    }]
                 }'
        """
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
        assert resp.json()["resultCode"] == "200"


# ============================================================
# 5. 任务控制接口
# ============================================================


class TestControlTask:
    """任务控制接口

    接口：POST /v1/service/controlTask
    功能：控制视频分析任务的启停和删除

    command 值:
        0 = 停止任务
        1 = 启动/恢复任务
        2 = 删除任务
    """

    def test_stop_task(self, client):
        """停止任务

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/controlTask \\
                 -H "Content-Type: application/json" \\
                 -d '{"analyseId": "stream-001", "command": 0}'
        """
        resp = client.post(
            "/v1/service/controlTask",
            json={"analyseId": "stream-001", "command": 0},
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "200"
        assert resp.json()["resultHint"] == "任务已停止"

    def test_resume_task(self, client):
        """恢复/启动任务

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/controlTask \\
                 -H "Content-Type: application/json" \\
                 -d '{"analyseId": "stream-001", "command": 1}'
        """
        mock_task = MagicMock()
        mock_task.is_running = False
        client._mock_ctrl_vtm.get_task.return_value = mock_task

        resp = client.post(
            "/v1/service/controlTask",
            json={"analyseId": "stream-001", "command": 1},
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "200"
        assert resp.json()["resultHint"] == "任务已启动"

    def test_delete_task(self, client):
        """删除任务

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/controlTask \\
                 -H "Content-Type: application/json" \\
                 -d '{"analyseId": "stream-001", "command": 2}'
        """
        resp = client.post(
            "/v1/service/controlTask",
            json={"analyseId": "stream-001", "command": 2},
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "200"
        assert resp.json()["resultHint"] == "任务已删除"

    def test_stop_nonexistent(self, client):
        """停止不存在的任务返回 404"""
        client._mock_ctrl_vtm.stop_task.return_value = False

        resp = client.post(
            "/v1/service/controlTask",
            json={"analyseId": "nonexistent", "command": 0},
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "404"

    def test_invalid_command(self, client):
        """无效的控制指令返回 400"""
        resp = client.post(
            "/v1/service/controlTask",
            json={"analyseId": "stream-001", "command": 99},
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "400"


# ============================================================
# 6. 保活心跳
# ============================================================


class TestKeepAlive:
    """保活心跳接口"""

    def test_keep_alive(self, client):
        """POST /analysis/api/v1/keepAlive — 保活心跳

        cURL:
            curl -X POST http://127.0.0.1:22266/analysis/api/v1/keepAlive \\
                 -H "Content-Type: application/json" \\
                 -d '{"devIP": "192.168.1.50", "devPort": 22266}'
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
# 7. 更新分析ID
# ============================================================


class TestUpdateAnalyseID:
    """更新分析ID接口"""

    def test_update_success(self, client):
        """POST /analysis/api/v1/updateAnalyseID — 更新分析ID（成功）

        cURL:
            curl -X POST http://127.0.0.1:22266/analysis/api/v1/updateAnalyseID \\
                 -H "Content-Type: application/json" \\
                 -d '{"oldAnalyseId": "old-001", "newAnalyseId": "new-001"}'
        """
        resp = client.post(
            "/analysis/api/v1/updateAnalyseID",
            json={"oldAnalyseId": "old-001", "newAnalyseId": "new-001"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"]["newAnalyseId"] == "new-001"

    def test_update_not_found(self, client):
        """更新不存在的分析ID返回 404"""
        client._mock_upd_vtm.update_analyse_id.return_value = False

        resp = client.post(
            "/analysis/api/v1/updateAnalyseID",
            json={"oldAnalyseId": "nonexistent", "newAnalyseId": "new-001"},
        )

        assert resp.status_code == 200
        assert resp.json()["resultCode"] == "404"


# ============================================================
# 8. 样本上传
# ============================================================


class TestUploadSamples:
    """样本上传接口"""

    def test_upload_sample(self, client):
        """POST /v1/service/uploadSamples — 上传样本

        cURL:
            curl -X POST http://127.0.0.1:22266/v1/service/uploadSamples \\
                 -H "Content-Type: application/json" \\
                 -d '{"fileId": "sample-001", "md5": "abc123def456"}'
        """
        resp = client.post(
            "/v1/service/uploadSamples",
            json={"fileId": "sample-001", "md5": "abc123def456"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resultCode"] == "200"
        assert data["resultValue"]["fileId"] == "sample-001"
