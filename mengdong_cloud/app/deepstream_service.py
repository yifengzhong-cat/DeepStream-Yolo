from __future__ import annotations

import base64
import hashlib
import os
import signal
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .model_catalog import MODEL_ABILITIES, get_ability_or_none


@dataclass
class TaskRecord:
    analyse_id: str
    dev_code: str
    alg_code: str
    video_url: str | None = None
    format_type: int | None = None
    status: str = "created"
    process: subprocess.Popen[str] | None = None
    osd_video_url: str | None = None
    runtime_config: Path | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class DeepStreamTaskService:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.template_config = repo_root / "deepstream_app_config.txt"
        self.runtime_dir = Path(os.getenv("MENGDONG_RUNTIME_DIR", "/tmp/mengdong_cloud"))
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.callback_results: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, TaskRecord] = {}
        self._lock = Lock()

    def ability_response(self) -> dict[str, Any]:
        abilities = []
        for ability in MODEL_ABILITIES.values():
            abilities.append(
                {
                    "algCode": ability.alg_code,
                    "algDesc": ability.alg_desc,
                    "algParams": [
                        {"key": "--sensitivity", "value": "灵敏度,范围[1,5]"},
                        {"key": "--confidence", "value": "置信度阈值,范围[0.1,1.0]"},
                    ],
                }
            )
        return {
            "resultCode": "200",
            "resultValue": {
                "abilityInfo": {
                    "number": len(abilities),
                    "ability": abilities,
                }
            },
            "resultHint": None,
        }

    def create_video_tasks(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        alg_code = str(payload.get("algCode", ""))
        command = int(payload.get("command", 1))
        ability = get_ability_or_none(alg_code)
        if ability is None:
            return 400, {"resultCode": "400", "resultValue": None, "resultHint": f"unknown algCode: {alg_code}"}

        videos = payload.get("videoInfo", []) or []
        if not isinstance(videos, list) or not videos:
            return 400, {"resultCode": "400", "resultValue": None, "resultHint": "videoInfo is required"}

        results = []
        with self._lock:
            for video in videos:
                analyse_id = str(video.get("analyseId", "")).strip()
                dev_code = str(video.get("devCode", "")).strip()
                video_url = str(video.get("videoUrl", "")).strip()
                format_type = int(video.get("formatType", 0))
                if not analyse_id or not dev_code or not video_url:
                    return 400, {"resultCode": "400", "resultValue": None, "resultHint": "analyseId/devCode/videoUrl required"}

                existing = self.tasks.get(analyse_id)
                if command == 2 and existing:
                    self._stop_task(existing, delete=True)
                elif command == 0 and existing:
                    self._stop_task(existing, delete=False)
                elif command == 1:
                    if existing:
                        self._stop_task(existing, delete=True)
                    task = TaskRecord(
                        analyse_id=analyse_id,
                        dev_code=dev_code,
                        alg_code=alg_code,
                        video_url=video_url,
                        format_type=format_type,
                        osd_video_url=f"rtsp://0.0.0.0:8554/{analyse_id}",
                    )
                    self._start_task(task, ability.infer_config)
                    self.tasks[analyse_id] = task

                results.append(
                    {
                        "analyseId": analyse_id,
                        "devCode": dev_code,
                        "osdVideoUrl": f"rtsp://0.0.0.0:8554/{analyse_id}",
                    }
                )

        return 200, {"resultCode": "200", "resultValue": results, "resultHint": None}

    def control_task(self, analyse_id: str, command: int) -> tuple[int, dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(analyse_id)
            if not task:
                return 404, {"resultCode": "404", "resultValue": None, "resultHint": "analyseId not found"}
            if command == 0:
                self._stop_task(task, delete=False)
            elif command == 1 and task.status in {"stopped", "mock_stopped"}:
                ability = get_ability_or_none(task.alg_code)
                if ability is None:
                    return 400, {"resultCode": "400", "resultValue": None, "resultHint": "unknown algCode"}
                self._start_task(task, ability.infer_config)
            elif command == 2:
                self._stop_task(task, delete=True)
            else:
                return 400, {"resultCode": "400", "resultValue": None, "resultHint": "unsupported command"}

        return 200, {"resultCode": "200", "resultValue": None, "resultHint": "operation success"}

    def image_task(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        analyse_id = str(payload.get("analyseId", "")).strip()
        alg_code = str(payload.get("algCode", "")).strip()
        image_data = str(payload.get("imageData", "")).strip()
        ability = get_ability_or_none(alg_code)
        if not analyse_id or not alg_code or not image_data:
            return 400, {"resultCode": "400", "resultValue": None, "resultHint": "analyseId/algCode/imageData required"}
        if ability is None:
            return 400, {"resultCode": "400", "resultValue": None, "resultHint": f"unknown algCode: {alg_code}"}

        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception:
            return 400, {"resultCode": "400", "resultValue": None, "resultHint": "imageData is not valid base64"}

        image_name = f"{analyse_id}.jpg"
        image_path = self.runtime_dir / image_name
        image_path.write_bytes(image_bytes)

        analyse_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        result_value = {
            "analyseResults": [],
            "analyseTime": analyse_time,
            "rawImageName": image_name,
            "rawImageData": image_data,
            "osdImageName": image_name,
            "osdImageData": image_data,
            "resultDetail": [],
        }
        self.callback_results[analyse_id] = {
            "algCode": alg_code,
            "analyseId": analyse_id,
            "analyseTime": analyse_time,
            "analyseResults": [],
            "rawImageName": image_name,
            "rawImageData": image_data,
            "osdImageName": image_name,
            "osdImageData": image_data,
            "resultDetail": [],
        }
        return 200, {"resultCode": "200", "resultValue": result_value, "resultHint": None}

    def upload_samples(self, file_id: str, md5_value: str) -> tuple[int, dict[str, Any]]:
        if not file_id or not md5_value:
            return 400, {"resultCode": "400", "resultValue": None, "resultHint": "fileId and md5 are required"}
        return (
            200,
            {
                "resultCode": "200",
                "resultValue": {"fileId": file_id},
                "resultHint": "接收样本成功",
            },
        )

    def keep_alive(self, dev_ip: str, dev_port: int) -> tuple[int, dict[str, Any]]:
        if not dev_ip or dev_port <= 0:
            return 400, {"resultCode": "400", "resultValue": None, "resultHint": "devIP/devPort invalid"}
        dev_hash = hashlib.sha256(f"{dev_ip}:{dev_port}".encode("utf-8")).hexdigest()[:24]
        return (
            200,
            {
                "resultCode": "200",
                "resultValue": {"devId": dev_hash},
                "resultHint": "keep alive",
            },
        )

    def update_analyse_id(self, old_analyse_id: str, new_analyse_id: str) -> tuple[int, dict[str, Any]]:
        if not old_analyse_id or not new_analyse_id:
            return 400, {"resultCode": "400", "resultValue": None, "resultHint": "oldAnalyseId/newAnalyseId required"}

        with self._lock:
            task = self.tasks.pop(old_analyse_id, None)
            if task:
                task.analyse_id = new_analyse_id
                task.osd_video_url = f"rtsp://0.0.0.0:8554/{new_analyse_id}"
                self.tasks[new_analyse_id] = task
            if old_analyse_id in self.callback_results:
                self.callback_results[new_analyse_id] = self.callback_results.pop(old_analyse_id)
                self.callback_results[new_analyse_id]["analyseId"] = new_analyse_id

        return (
            200,
            {
                "resultCode": "200",
                "resultValue": {"newAnalyseId": new_analyse_id},
                "resultHint": "keep alive",
            },
        )

    def save_callback_result(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        analyse_id = str(payload.get("analyseId", "")).strip()
        if not analyse_id:
            return 400, {"resultCode": "400", "resultHint": "analyseId required"}
        with self._lock:
            self.callback_results[analyse_id] = payload
        return 200, {"resultCode": "200", "resultHint": ""}

    def _start_task(self, task: TaskRecord, infer_config: str) -> None:
        deepstream_app = shutil.which("deepstream-app")
        if deepstream_app is None:
            task.status = "mock_running"
            task.process = None
            return

        run_config = self._prepare_runtime_config(task.video_url or "", infer_config)
        task.runtime_config = run_config
        task.process = subprocess.Popen(
            [deepstream_app, "-c", str(run_config)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        task.status = "running"

    def _stop_task(self, task: TaskRecord, delete: bool) -> None:
        if task.process and task.process.poll() is None:
            task.process.terminate()
            try:
                task.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                task.process.send_signal(signal.SIGKILL)
                task.process.wait(timeout=3)

        if task.runtime_config and task.runtime_config.exists():
            task.runtime_config.unlink(missing_ok=True)
            task.runtime_config = None

        if delete:
            self.tasks.pop(task.analyse_id, None)
            self.callback_results.pop(task.analyse_id, None)
        else:
            if task.status.startswith("mock"):
                task.status = "mock_stopped"
            else:
                task.status = "stopped"

    def _prepare_runtime_config(self, video_url: str, infer_config: str) -> Path:
        runtime_file = Path(tempfile.mkstemp(prefix="deepstream_", suffix=".txt", dir=self.runtime_dir)[1])
        content = self.template_config.read_text(encoding="utf-8")
        lines = []
        in_source0 = False
        in_primary_gie = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_source0 = stripped == "[source0]"
                in_primary_gie = stripped == "[primary-gie]"
                lines.append(line)
                continue
            if in_source0 and stripped.startswith("uri="):
                lines.append(f"uri={video_url}")
                continue
            if in_primary_gie and stripped.startswith("config-file="):
                lines.append(f"config-file={infer_config}")
                continue
            lines.append(line)
        runtime_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return runtime_file


service = DeepStreamTaskService(Path(__file__).resolve().parents[2])
