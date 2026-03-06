# MengDong Cloud API（基于 DeepStream-Yolo）

该目录提供一个独立的 FastAPI 服务层，用于在 **DeepStream-Yolo** 基础上开放图片、视频、视频流任务接口。

## 全流程文档（中文）

- 请先阅读：[`OPERATION_GUIDE_CN.md`](OPERATION_GUIDE_CN.md)
- 覆盖内容：
  - `pt -> onnx -> engine` 转换与校验
  - INT8 校准与 INT8 engine 生成
  - FastAPI 服务启动
  - `curl` 命令联调（abilities / videoTask / controlTask / imageTask）
  - 实际视频流推理并保存为 `mp4`

## 启动

```bash
cd mengdong_cloud
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 22266
```

## 已实现接口

- `POST /v1/service/abilities`
- `POST /v1/service/videoTask`
- `POST /v1/service/controlTask`
- `POST /v1/service/imageTask`
- `POST /v1/service/uploadSamples`
- `POST /analysis/api/v1/analyseResult`
- `POST /analysis/api/v1/keepAlive`
- `POST /analysis/api/v1/updateAnalyseID`

## 四个模型映射（algCode）

- `101001`：人员穿戴识别（person/safetybelt/mapblu/mapor/mapr/mapwh/mapy）
- `101002`：hook/SRL
- `101003`：tower
- `101004`：AerialWorkBucketTruck/ConductorSpacer/byq/daozha/dlq/gk/jsj/pcs/pole/xl

> 说明：接口服务会优先尝试调用 `deepstream-app`。若运行环境尚未安装 DeepStream，可先完成接口联调，任务会进入 mock 运行状态。
