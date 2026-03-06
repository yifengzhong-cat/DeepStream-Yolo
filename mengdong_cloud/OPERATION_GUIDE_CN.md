# MengDong Cloud 全流程操作文档（中文）

本文档用于打通以下完整流程：

1. 将 YOLO `*.pt` 模型转换到 DeepStream 可用的 `*.engine`
2. 启动 `mengdong_cloud` 接口服务
3. 使用命令行（`curl`）测试接口
4. 对真实视频流进行推理，并将结果保存为 `mp4`

---

## 1. 前置准备

在仓库根目录执行（示例路径）：

```bash
export REPO_ROOT=/home/runner/work/DeepStream-Yolo/DeepStream-Yolo
cd "${REPO_ROOT}"
```

确保以下文件已准备好：

- 模型配置：`config_infer_primary_yoloV8.txt`
- 标签文件：`labels.txt`
- 解析库：`nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`

---

## 2. PT 模型转换为 ENGINE（INT8 流程，本文重点）

> DeepStream 不直接读取 `pt`，通常是 `pt -> onnx -> engine`。
> 本仓库已提供 YOLOv8 导出脚本：`utils/export_yoloV8.py`。
> 本节聚焦 INT8。如需 FP16/FP32，可参考仓库 `docs/YOLOv8.md` 与 `docs/customModels.md`。

### 2.1 导出 ONNX

在 Ultralytics 环境中执行（示例）：

```bash
python3 export_yoloV8.py -w your_model.pt --dynamic --simplify
```

生成 `your_model.onnx` 后，拷贝到仓库根目录，例如：

```bash
cp your_model.onnx "${REPO_ROOT}/"
```

### 2.2 修改推理配置（INT8）

编辑 `${REPO_ROOT}/config_infer_primary_yoloV8.txt`：

- `onnx-file=your_model.onnx`
- `model-engine-file=your_model_b1_gpu0_int8.engine`
- `int8-calib-file=calib.table`
- `num-detected-classes=<你的类别数>`
- `network-mode=1`（INT8）

### 2.3 准备 INT8 校准数据（PTQ）

在仓库根目录准备校准数据清单（示例）：

```bash
mkdir -p calibration
# 放入至少 500 张，建议 1000 张与你的业务分布接近的图片到 calibration/
find "${REPO_ROOT}/calibration" -type f -name "*.jpg" -print | sort > "${REPO_ROOT}/calibration.txt"
```

> 推荐使用 `.jpg`。若你的数据是 `.jpeg`/`.png`，请先统一转为 `.jpg` 再生成 `calibration.txt`。  
> 少于 500 张也可运行，但通常会导致量化误差更大、INT8 精度更不稳定。

设置校准环境变量（DeepStream 读取）：

```bash
export INT8_CALIB_IMG_PATH="${REPO_ROOT}/calibration.txt"
export INT8_CALIB_BATCH_SIZE=1
```

> 如需更快/更稳定的校准，请根据显存提高 `INT8_CALIB_BATCH_SIZE`。  
> 若未安装 OpenCV，请参考：`${REPO_ROOT}/docs/INT8Calibration.md`

### 2.4 生成 INT8 ENGINE（两种方式）

方式 A（推荐）：第一次运行 `deepstream-app` 自动生成：

```bash
deepstream-app -c "${REPO_ROOT}/deepstream_app_config.txt"
```

方式 B：使用 `trtexec` 手工生成（仅在你已具备 `calib.table` 时）：

```bash
trtexec --onnx=your_model.onnx \
  --int8 \
  --calib="${REPO_ROOT}/calib.table" \
  --saveEngine=your_model_b1_gpu0_int8.engine
```

> 如果你还没有 `calib.table`，建议优先使用方式 A（DeepStream 自动完成校准并生成 INT8 engine）。
> 生成成功后，确认 `model-engine-file` 指向的 engine 文件存在。

### 2.5 不走 Python，直接用 C++ 可以吗？

可以。`deepstream-app` 本身就是 C/C++ 应用，完成上述 ONNX/INT8 engine 准备后可直接运行：

```bash
deepstream-app -c "${REPO_ROOT}/deepstream_app_config.txt"
```

> 本文档第 3 章使用 Python（FastAPI）只是为了对外提供 HTTP 接口；底层推理仍是 DeepStream（C/C++ + TensorRT）。

---

## 3. 启动 mengdong_cloud 服务

```bash
cd "${REPO_ROOT}/mengdong_cloud"
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 22266
```

服务地址：`http://127.0.0.1:22266`

---

## 4. 命令行测试接口（curl）

## 4.1 查询能力

```bash
curl -s -X POST http://127.0.0.1:22266/v1/service/abilities \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 4.2 创建视频流任务（videoTask）

```bash
curl -s -X POST http://127.0.0.1:22266/v1/service/videoTask \
  -H "Content-Type: application/json" \
  -d '{
    "algCode":"101003",
    "command":1,
    "videoInfo":[
      {
        "analyseId":"tower-001",
        "devCode":"dev-001",
        "videoUrl":"rtsp://<camera-ip>/live/stream",
        "formatType":0
      }
    ]
  }'
```

返回中的 `osdVideoUrl` 为推理输出流地址，格式类似：

```text
rtsp://0.0.0.0:8554/tower-001
```

### 4.3 停止任务（controlTask）

```bash
curl -s -X POST http://127.0.0.1:22266/v1/service/controlTask \
  -H "Content-Type: application/json" \
  -d '{"analyseId":"tower-001","command":0}'
```

### 4.4 图片推理（imageTask）

```bash
IMG_BASE64=$(base64 -w 0 test.jpg)
curl -s -X POST http://127.0.0.1:22266/v1/service/imageTask \
  -H "Content-Type: application/json" \
  -d "{\"analyseId\":\"img-001\",\"algCode\":\"101003\",\"imageData\":\"${IMG_BASE64}\"}"
```

---

## 5. 实际视频流推理并保存为 MP4

当前 `videoTask` 默认返回 RTSP 推理流地址。最稳妥做法是使用 `ffmpeg` 将该 RTSP 输出保存为 MP4。

假设 `osdVideoUrl=rtsp://127.0.0.1:8554/tower-001`，执行：

```bash
ffmpeg -rtsp_transport tcp -i rtsp://127.0.0.1:8554/tower-001 \
  -c:v libx264 -pix_fmt yuv420p -an \
  -movflags +faststart \
  "${REPO_ROOT}/mengdong_cloud/output/tower-001.mp4"
```

如果希望“只封装不转码”（降低 CPU）且输入编码兼容 MP4，可尝试：

```bash
ffmpeg -rtsp_transport tcp -i rtsp://127.0.0.1:8554/tower-001 \
  -c copy -an \
  "${REPO_ROOT}/mengdong_cloud/output/tower-001.mp4"
```

> 建议先 `mkdir -p "${REPO_ROOT}/mengdong_cloud/output"`

---

## 6. 常见问题

1. **为什么 `pt` 不能直接推理？**  
   DeepStream 推理链路基于 TensorRT，需先转 ONNX/ENGINE。

2. **ENGINE 一直没生成？**  
   检查 `config_infer_primary_yoloV8.txt` 中的 `onnx-file` 路径、类别数、网络模式是否匹配；首次生成可能耗时较长。

3. **服务返回成功但没有真实推理？**  
   若运行环境找不到 `deepstream-app`，服务会进入 mock 状态。请确认 DeepStream 安装与 `deepstream-app` 可执行。

4. **为什么偏要用 Python，C++ 不行吗？**  
   可以不用 Python。模型导出阶段常用 Python 工具（`pt -> onnx`），但推理执行可直接使用 `deepstream-app`（C/C++）。只有在你需要 REST 接口时，才需要运行 `mengdong_cloud` 的 Python 服务层。
