# 蒙东云端AI分析平台 - 部署文档

## 1. 概述

蒙东云端AI分析平台是基于 YOLOv8 的云端AI推理服务，支持视频流分析和图片分析。服务以 Docker 容器方式部署，使用 GPU 加速推理，通过 RESTful API 提供服务。

### 1.1 支持的模型

| 模型文件 | 算法编码 (algCode) | 功能描述 | 检测标签 |
|---------|-------------------|---------|---------|
| `model_mengdong_raa_adjusted.pt` | 010101 | 人员穿戴检测 | person(人), safetybelt(安全带), mapblu(蓝色安全帽), mapor(橘色安全帽), mapr(红色安全帽), mapwh(白色安全帽), mapy(黄色安全帽) |
| `model_mengdong_small_SRL.pt` | 010102 | 钩子与差速器检测 | hook(钩子), SRL(差速器) |
| `model_mengdong_tower.pt` | 010103 | 铁塔检测 | tower(铁塔) |
| `model_mengdong_scene_album.pt` | 010104 | 场景设备检测 | AerialWorkBucketTruck(斗臂车), ConductorSpacer(导线间隔棒), byq(变压器), daozha(刀闸), dlq(断路器), gk(高空屋顶), jsj(脚手架), pcs(脚扣), pole(电线杆), xl(线路) |

### 1.2 系统要求

- Docker 19.03+
- NVIDIA Docker Runtime（nvidia-docker2）
- NVIDIA GPU（算力 6.0+，推荐 Tesla T4 / V100 / A100）
- NVIDIA 驱动 470+
- CUDA 12.1+

---

## 2. 快速部署

### 2.1 方式一：使用预构建镜像包（推荐）

```bash
# 1. 加载 Docker 镜像
docker load -i mengdong_cloud.tar

# 2. 启动容器
docker run -d \
  --gpus all \
  -p 22266:22266 \
  --name mengdong_cloud \
  --restart unless-stopped \
  mengdong_cloud:latest

# 3. 验证服务
curl http://localhost:22266/health
```

### 2.2 方式二：源码构建

```bash
# 1. 进入项目目录
cd mengdong_cloud

# 2. 将模型文件放入 models/ 目录
cp /path/to/model_mengdong_raa_adjusted.pt models/
cp /path/to/model_mengdong_small_SRL.pt models/
cp /path/to/model_mengdong_tower.pt models/
cp /path/to/model_mengdong_scene_album.pt models/

# 3. 构建并导出镜像（自动生成 mengdong_cloud.tar）
bash build.sh

# 4. 或使用 docker-compose 直接启动
docker-compose up -d
```

### 2.3 环境变量配置

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `SERVICE_HOST` | `0.0.0.0` | 服务监听地址 |
| `SERVICE_PORT` | `22266` | 服务监听端口 |
| `PLATFORM_HOST` | `http://127.0.0.1:8080` | 统一视频平台地址（用于结果上报和保活） |
| `KEEPALIVE_INTERVAL` | `30` | 保活心跳间隔（秒） |
| `DEVICE` | `0` | GPU 设备号（`0` 表示第一块 GPU，`cpu` 表示使用 CPU） |
| `CONFIDENCE_THRESHOLD` | `0.5` | 默认推理置信度阈值 |
| `MODEL_DIR` | `/app/models` | 模型文件目录 |
| `OUTPUT_DIR` | `/app/output` | 输出视频文件目录 |

示例：自定义环境变量启动

```bash
docker run -d \
  --gpus all \
  -p 22266:22266 \
  -e PLATFORM_HOST=http://192.168.1.100:8080 \
  -e CONFIDENCE_THRESHOLD=0.6 \
  -e KEEPALIVE_INTERVAL=60 \
  --name mengdong_cloud \
  mengdong_cloud:latest
```

---

## 3. API 接口文档

服务启动后可访问自动生成的交互式 API 文档：
- **Swagger UI**: `http://<服务器IP>:22266/docs`
- **ReDoc**: `http://<服务器IP>:22266/redoc`

### 3.1 健康检查

- **URL**: `GET /health`
- **说明**: 检查服务状态和模型加载情况

**响应示例**:
```json
{
    "status": "ok",
    "models_loaded": true,
    "loaded_count": 4
}
```

---

### 3.2 算法能力获取 (4.1.7.1)

- **URL**: `POST /v1/service/abilities`
- **说明**: 获取当前服务支持的所有算法能力列表

**请求**: 无请求体

**响应示例**:
```json
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
                        {
                            "key": "--sensitivity",
                            "value": "灵敏度,范围[1,5]"
                        }
                    ]
                },
                {
                    "algCode": "010102",
                    "algDesc": "钩子与差速器检测",
                    "algParams": [
                        {
                            "key": "--sensitivity",
                            "value": "灵敏度,范围[1,5]"
                        }
                    ]
                },
                {
                    "algCode": "010103",
                    "algDesc": "铁塔检测",
                    "algParams": [
                        {
                            "key": "--sensitivity",
                            "value": "灵敏度,范围[1,5]"
                        }
                    ]
                },
                {
                    "algCode": "010104",
                    "algDesc": "场景设备检测",
                    "algParams": [
                        {
                            "key": "--sensitivity",
                            "value": "灵敏度,范围[1,5]"
                        }
                    ]
                }
            ]
        }
    },
    "resultHint": null
}
```

---

### 3.3 视频任务管理 (4.1.7.2)

- **URL**: `POST /v1/service/videoTask`
- **说明**: 创建、启动、停止或删除视频分析任务

**请求示例**:
```json
{
    "algCode": "010101",
    "startTime": "2024-01-01 00:00:00",
    "endTime": "2024-12-31 23:59:59",
    "interval": 60,
    "command": 1,
    "videoInfo": [
        {
            "analyseId": "task_001",
            "devCode": "CAM_001",
            "formatType": 0,
            "videoUrl": "rtsp://192.168.1.100:554/stream1"
        }
    ],
    "rule": {
        "algParams": [
            {
                "key": "--sensitivity",
                "value": "3"
            }
        ],
        "ruleNum": 1,
        "ruleProperty": [
            {
                "ruleId": 0,
                "ruleDesc": "检测区域",
                "pointNum": 2,
                "point": [
                    {"id": 0, "pointDesc": "", "x": 0, "y": 0},
                    {"id": 1, "pointDesc": "", "x": 1920, "y": 1080}
                ]
            }
        ]
    }
}
```

**请求参数**:

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| algCode | String | 是 | 算法能力编码 |
| startTime | String | 是 | 任务开始时间 |
| endTime | String | 是 | 任务结束时间 |
| interval | Integer | 是 | 结果上送间隔（秒） |
| command | Integer | 是 | 0=停止, 1=开始, 2=删除 |
| videoInfo | Array | 是 | 视频信息列表 |
| videoInfo.analyseId | String | 是 | 分析任务ID |
| videoInfo.devCode | String | 是 | 摄像头设备编码 |
| videoInfo.formatType | Integer | 是 | 0=H264, 1=H265 |
| videoInfo.videoUrl | String | 是 | 视频流地址 |
| rule | Object | 否 | 规则框详情 |

**响应示例**:
```json
{
    "resultCode": "200",
    "resultValue": [
        {
            "analyseId": "task_001",
            "devCode": "CAM_001",
            "osdVideoUrl": "http://0.0.0.0:22266/output/task_001.mp4"
        }
    ],
    "resultHint": null
}
```

---

### 3.4 分析任务控制 (4.1.7.3)

- **URL**: `POST /v1/service/controlTask`
- **说明**: 控制已创建的分析任务（启停、删除）

**请求示例**:
```json
{
    "analyseId": "task_001",
    "command": 0
}
```

**请求参数**:

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| analyseId | String | 是 | 分析任务ID |
| command | Integer | 是 | 0=停止, 1=开始, 2=删除 |

**响应示例**:
```json
{
    "resultCode": "200",
    "resultValue": null,
    "resultHint": "任务已停止"
}
```

---

### 3.5 图片分析任务 (4.1.7.5)

- **URL**: `POST /v1/service/imageTask`
- **说明**: 对单张图片进行推理分析，同步返回结果

**请求示例**:
```json
{
    "analyseId": "img_001",
    "algCode": "010101",
    "imageData": "<Base64编码的JPG图片>",
    "rule": {
        "algParams": [
            {
                "key": "--sensitivity",
                "value": "3"
            }
        ]
    }
}
```

**请求参数**:

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| analyseId | String | 是 | 分析ID |
| algCode | String | 是 | 算法编码 |
| imageData | String | 是 | Base64 编码的 JPG 图片数据 |
| rule | Object | 否 | 规则框详情（含灵敏度参数） |

**响应示例**:
```json
{
    "resultCode": "200",
    "resultValue": {
        "analyseResults": ["人", "白色安全帽"],
        "analyseTime": "2024-01-15 10:30:00",
        "rawImageName": "img_001_raw.jpg",
        "rawImageData": "<Base64编码原图>",
        "osdImageName": "img_001_osd.jpg",
        "osdImageData": "<Base64编码标注图>",
        "resultDetail": [
            {
                "algCode": "010101",
                "resultDesc": "人",
                "num": 2,
                "resultItems": [
                    {
                        "score": 95.7,
                        "leftTopX": 100,
                        "leftTopY": 50,
                        "rightBottomX": 300,
                        "rightBottomY": 400
                    },
                    {
                        "score": 88.3,
                        "leftTopX": 500,
                        "leftTopY": 100,
                        "rightBottomX": 700,
                        "rightBottomY": 450
                    }
                ]
            },
            {
                "algCode": "010101",
                "resultDesc": "白色安全帽",
                "num": 1,
                "resultItems": [
                    {
                        "score": 92.1,
                        "leftTopX": 120,
                        "leftTopY": 30,
                        "rightBottomX": 200,
                        "rightBottomY": 80
                    }
                ]
            }
        ]
    },
    "resultHint": null
}
```

---

### 3.6 视频分析结果推送 (4.1.7.4)

- **说明**: 视频分析过程中，服务会按照配置的 `interval` 间隔，自动向统一视频平台推送分析结果
- **推送目标**: `{PLATFORM_HOST}/analysis/api/v1/analyseResult`
- **推送方式**: POST

**推送数据格式**:
```json
{
    "algCode": "010101",
    "analyseId": "task_001",
    "analyseTime": "2024-01-15 10:30:00",
    "analyseResults": ["未带安全帽"],
    "rawImageName": "task_001_raw.jpg",
    "rawImageData": "<Base64编码原图>",
    "osdImageName": "task_001_osd.jpg",
    "osdImageData": "<Base64编码标注图>",
    "resultDetail": [
        {
            "algCode": "010101",
            "resultDesc": "未带安全帽",
            "num": 1,
            "resultItems": [
                {
                    "score": 93.5,
                    "leftTopX": 200,
                    "leftTopY": 100,
                    "rightBottomX": 400,
                    "rightBottomY": 350
                }
            ]
        }
    ]
}
```

---

### 3.7 服务保活 (4.1.7.7)

- **URL**: `POST /analysis/api/v1/keepAlive`
- **说明**: 接收保活心跳（服务同时会定时向平台发送保活）

**请求示例**:
```json
{
    "devIP": "192.168.1.50",
    "devPort": 22266
}
```

**响应示例**:
```json
{
    "resultCode": "200",
    "resultValue": {
        "devId": "192.168.1.50:22266"
    },
    "resultHint": "keep alive"
}
```

---

### 3.8 更新分析ID (4.1.7.8)

- **URL**: `POST /analysis/api/v1/updateAnalyseID`
- **说明**: 更新已有任务的分析ID，自动重新拉流

**请求示例**:
```json
{
    "oldAnalyseId": "task_001",
    "newAnalyseId": "task_001_new"
}
```

**响应示例**:
```json
{
    "resultCode": "200",
    "resultValue": {
        "newAnalyseId": "task_001_new"
    },
    "resultHint": "更新成功"
}
```

---

### 3.9 样本数据回传 (4.1.7.6)

- **URL**: `POST /v1/service/uploadSamples`
- **说明**: 接收样本文件信息

**请求示例**:
```json
{
    "fileId": "file_001",
    "md5": "abc123def456"
}
```

**响应示例**:
```json
{
    "resultCode": "200",
    "resultValue": {
        "fileId": "file_001"
    },
    "resultHint": "接收样本成功"
}
```

---

## 4. 响应码说明

| 响应码 | 说明 |
|--------|------|
| 200 | 响应成功 |
| 400 | 消息格式错误 |
| 403 | 请求被禁止，无权限 |
| 404 | 请求的对象不存在 |
| 500 | 服务异常 |
| 503 | 当前负荷满，稍后再尝试 |

---

## 5. 部署运维

### 5.1 查看日志

```bash
docker logs -f mengdong_cloud
```

### 5.2 重启服务

```bash
docker restart mengdong_cloud
```

### 5.3 停止服务

```bash
docker stop mengdong_cloud
```

### 5.4 删除容器

```bash
docker stop mengdong_cloud
docker rm mengdong_cloud
```

### 5.5 更新镜像

```bash
# 停止并删除旧容器
docker stop mengdong_cloud
docker rm mengdong_cloud

# 加载新镜像
docker load -i mengdong_cloud.tar

# 启动新容器
docker run -d --gpus all -p 22266:22266 --name mengdong_cloud mengdong_cloud:latest
```

### 5.6 挂载外部模型目录

如果需要替换模型文件而不重新构建镜像：

```bash
docker run -d \
  --gpus all \
  -p 22266:22266 \
  -v /path/to/local/models:/app/models \
  --name mengdong_cloud \
  mengdong_cloud:latest
```

### 5.7 查看输出视频

视频分析任务产生的标注视频保存在容器内 `/app/output/` 目录，也可通过 HTTP 直接下载：

```bash
# 通过 HTTP 下载
curl -O http://<服务器IP>:22266/output/<analyseId>.mp4

# 或挂载输出目录
docker run -d \
  --gpus all \
  -p 22266:22266 \
  -v /path/to/local/output:/app/output \
  --name mengdong_cloud \
  mengdong_cloud:latest
```

---

## 6. 常见问题

### Q1: 服务启动后提示模型加载失败

**A**: 请检查模型文件是否正确放置在 `/app/models/` 目录下，文件名必须与配置一致：
- `model_mengdong_raa_adjusted.pt`
- `model_mengdong_small_SRL.pt`
- `model_mengdong_tower.pt`
- `model_mengdong_scene_album.pt`

### Q2: GPU 不可用

**A**: 确认已安装 NVIDIA Docker Runtime：
```bash
# 检查 nvidia-docker
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

如果 GPU 不可用，可设置环境变量 `DEVICE=cpu` 使用 CPU 推理（速度较慢）。

### Q3: 视频流无法拉取

**A**: 检查视频流地址是否可达。在容器内测试：
```bash
docker exec mengdong_cloud python -c "import cv2; cap=cv2.VideoCapture('rtsp://...'); print(cap.isOpened())"
```

### Q4: 如何调整推理灵敏度

**A**: 通过 API 请求中的 `rule.algParams` 设置灵敏度：
```json
{
    "algParams": [
        {
            "key": "--sensitivity",
            "value": "3"
        }
    ]
}
```
灵敏度范围 1-5，数值越大检测越灵敏（置信度阈值越低）。

### Q5: 端口冲突

**A**: 修改端口映射：
```bash
docker run -d --gpus all -p 8888:22266 --name mengdong_cloud mengdong_cloud:latest
```

---

## 7. 目录结构

```
mengdong_cloud/
├── Dockerfile              # Docker 镜像构建文件
├── docker-compose.yml      # Docker Compose 编排文件
├── requirements.txt        # Python 依赖
├── build.sh                # 构建和导出脚本
├── README_DEPLOY.md        # 本部署文档
├── app/                    # 应用代码
│   ├── main.py             # FastAPI 主入口
│   ├── config.py           # 配置文件
│   ├── schemas.py          # 请求/响应数据模型
│   ├── model_manager.py    # 模型管理器
│   ├── inference.py        # 推理辅助函数
│   ├── routers/            # API 路由
│   │   ├── abilities.py    # 算法能力接口
│   │   ├── video_task.py   # 视频任务接口
│   │   ├── control_task.py # 任务控制接口
│   │   ├── image_task.py   # 图片分析接口
│   │   ├── keep_alive.py   # 保活接口
│   │   ├── update_analyse.py # 更新分析ID接口
│   │   └── upload_samples.py # 样本上传接口
│   └── tasks/              # 后台任务
│       └── video_processor.py # 视频流处理器
├── models/                 # 模型文件目录
│   ├── model_mengdong_raa_adjusted.pt
│   ├── model_mengdong_small_SRL.pt
│   ├── model_mengdong_tower.pt
│   └── model_mengdong_scene_album.pt
└── output/                 # 输出视频目录
```
