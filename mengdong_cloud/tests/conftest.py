# tests/conftest.py
#
# 测试共享配置和 fixtures。
#
# 关键：必须在所有 app 模块导入之前强制设置环境变量，
# 以避免在 Docker 容器中（MODEL_DIR=/app/models 等已设置）或
# 非 root 环境中访问 /app 路径导致 PermissionError。

import base64
import os
import sys
import tempfile

# ----------------------------------------------------------------
# 1. 在任何 app 模块被导入之前，强制设置环境变量到临时目录
# ----------------------------------------------------------------
_tmpdir = tempfile.mkdtemp(prefix="mengdong_test_")

# 必须使用 os.environ[key] = value 而不是 setdefault，
# 否则在 Docker 容器中已设置的环境变量不会被覆盖
os.environ["MODEL_DIR"] = os.path.join(_tmpdir, "models")
os.environ["OUTPUT_DIR"] = os.path.join(_tmpdir, "output")
os.environ["DEEPSTREAM_CONFIG_DIR"] = os.path.join(_tmpdir, "ds_configs")
os.environ["DEEPSTREAM_YOLO_DIR"] = os.path.join(_tmpdir, "deepstream_yolo")

# 确保临时目录存在
for d in [
    os.environ["MODEL_DIR"],
    os.environ["OUTPUT_DIR"],
    os.environ["DEEPSTREAM_CONFIG_DIR"],
    os.environ["DEEPSTREAM_YOLO_DIR"],
]:
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------
# 2. 如果 app.config 已经被导入过，需要强制重新加载
#    以确保新的环境变量生效
# ----------------------------------------------------------------
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("app."):
        del sys.modules[mod_name]

# ----------------------------------------------------------------
# 3. 现在可以安全导入 app 模块了
# ----------------------------------------------------------------
from unittest.mock import MagicMock, patch  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.inference import Detection  # noqa: E402


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------


@pytest.fixture()
def sample_image_base64():
    """生成一个简单的 100x100 蓝色 JPEG 图片，返回 Base64 字符串。"""
    import cv2

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # 蓝色 BGR
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


@pytest.fixture()
def sample_detections():
    """模拟 DeepStream 推理返回的检测结果（algCode=010101 人员穿戴检测）。

    class_id=0 → 人
    class_id=6 → 黄色安全帽
    """
    return [
        Detection(class_id=0, confidence=0.92, x1=10, y1=20, x2=50, y2=80),
        Detection(class_id=6, confidence=0.85, x1=15, y1=10, x2=45, y2=35),
    ]


@pytest.fixture()
def client():
    """创建 FastAPI TestClient。

    同时 mock 掉 model_manager 和 video_task_manager，
    使测试脱离 GPU/DeepStream 依赖。
    """
    with (
        patch("app.main.model_manager") as mock_mm,
        patch("app.routers.image_task.model_manager") as mock_img_mm,
        patch("app.routers.video_task.video_task_manager") as mock_vtm,
        patch("app.routers.control_task.video_task_manager") as mock_ctrl_vtm,
        patch("app.routers.update_analyse.video_task_manager") as mock_upd_vtm,
    ):
        # model_manager（main 模块和 image_task 模块各有一个引用）
        for mm in (mock_mm, mock_img_mm):
            mm.is_loaded = True
            mm.loaded_models = {
                "010101": {}, "010102": {}, "010103": {}, "010104": {},
            }
            mm.ds_configs = {
                "010101": {"infer_config_path": "/tmp/fake.txt"},
                "010102": {"infer_config_path": "/tmp/fake.txt"},
                "010103": {"infer_config_path": "/tmp/fake.txt"},
                "010104": {"infer_config_path": "/tmp/fake.txt"},
            }
            mm.load_all_models = MagicMock()
            mm.get_ds_config = MagicMock(
                side_effect=lambda code: {
                    "010101": {"infer_config_path": "/tmp/fake.txt"},
                    "010102": {"infer_config_path": "/tmp/fake.txt"},
                    "010103": {"infer_config_path": "/tmp/fake.txt"},
                    "010104": {"infer_config_path": "/tmp/fake.txt"},
                }.get(code)
            )
            mm.predict = MagicMock(return_value=[])

        # video_task_manager
        for vtm in (mock_vtm, mock_ctrl_vtm, mock_upd_vtm):
            vtm.create_task = MagicMock(return_value=MagicMock())
            vtm.stop_task = MagicMock(return_value=True)
            vtm.delete_task = MagicMock(return_value=True)
            vtm.get_task = MagicMock(return_value=None)
            vtm.update_analyse_id = MagicMock(return_value=True)

        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as tc:
            # 把 mock 对象也暴露出来，方便测试中进一步配置
            tc._mock_mm = mock_img_mm
            tc._mock_vtm = mock_vtm
            tc._mock_ctrl_vtm = mock_ctrl_vtm
            tc._mock_upd_vtm = mock_upd_vtm
            yield tc
