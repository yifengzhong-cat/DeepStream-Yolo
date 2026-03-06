from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .deepstream_service import service


app = FastAPI(title="MengDong DeepStream Cloud Service", version="1.0.0")


class ControlTaskRequest(BaseModel):
    analyseId: str = Field(..., min_length=1)
    command: int = Field(..., ge=0, le=2)


class UploadSamplesRequest(BaseModel):
    fileId: str = Field(..., min_length=1)
    md5: str = Field(..., min_length=1)


class KeepAliveRequest(BaseModel):
    devIP: str = Field(..., min_length=1)
    devPort: int = Field(..., gt=0)


class UpdateAnalyseIdRequest(BaseModel):
    oldAnalyseId: str = Field(..., min_length=1)
    newAnalyseId: str = Field(..., min_length=1)


@app.post("/v1/service/abilities")
async def abilities() -> dict:
    return service.ability_response()


@app.post("/v1/service/videoTask")
async def video_task(request: Request) -> JSONResponse:
    payload = await request.json()
    status_code, body = service.create_video_tasks(payload)
    return JSONResponse(content=body, status_code=status_code)


@app.post("/v1/service/controlTask")
async def control_task(payload: ControlTaskRequest) -> JSONResponse:
    status_code, body = service.control_task(payload.analyseId, payload.command)
    return JSONResponse(content=body, status_code=status_code)


@app.post("/v1/service/imageTask")
async def image_task(request: Request) -> JSONResponse:
    payload = await request.json()
    status_code, body = service.image_task(payload)
    return JSONResponse(content=body, status_code=status_code)


@app.post("/v1/service/uploadSamples")
async def upload_samples(payload: UploadSamplesRequest) -> JSONResponse:
    status_code, body = service.upload_samples(payload.fileId, payload.md5)
    return JSONResponse(content=body, status_code=status_code)


@app.post("/analysis/api/v1/analyseResult")
async def analyse_result(request: Request) -> JSONResponse:
    payload = await request.json()
    status_code, body = service.save_callback_result(payload)
    return JSONResponse(content=body, status_code=status_code)


@app.post("/analysis/api/v1/keepAlive")
async def keep_alive(payload: KeepAliveRequest) -> JSONResponse:
    status_code, body = service.keep_alive(payload.devIP, payload.devPort)
    return JSONResponse(content=body, status_code=status_code)


@app.post("/analysis/api/v1/updateAnalyseID")
async def update_analyse_id(payload: UpdateAnalyseIdRequest) -> JSONResponse:
    status_code, body = service.update_analyse_id(payload.oldAnalyseId, payload.newAnalyseId)
    return JSONResponse(content=body, status_code=status_code)
