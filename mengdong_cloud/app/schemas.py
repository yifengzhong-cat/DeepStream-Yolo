# Pydantic 请求/响应模型定义

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==================== 通用 ====================

class AlgParam(BaseModel):
    key: str = ""
    value: str = ""


class Point(BaseModel):
    id: Optional[int] = 0
    pointDesc: Optional[str] = ""
    x: Optional[int] = 0
    y: Optional[int] = 0


class RuleProperty(BaseModel):
    ruleId: Optional[int] = 0
    ruleDesc: Optional[str] = ""
    pointNum: Optional[int] = 0
    point: Optional[List[Point]] = []


class Rule(BaseModel):
    algParams: Optional[List[AlgParam]] = []
    ruleNum: Optional[int] = 0
    ruleProperty: Optional[List[RuleProperty]] = []


# ==================== 4.1.7.1 算法能力 ====================

class AbilityItem(BaseModel):
    algCode: str
    algDesc: str = ""
    algParams: Optional[List[AlgParam]] = []


class AbilityInfo(BaseModel):
    number: int
    ability: List[AbilityItem]


class AbilitiesResponse(BaseModel):
    resultCode: str = "200"
    resultValue: Optional[Dict[str, Any]] = None
    resultHint: Optional[str] = None


# ==================== 4.1.7.2 视频任务 ====================

class VideoInfo(BaseModel):
    analyseId: str
    devCode: str = ""
    formatType: int = 0
    videoUrl: str


class VideoTaskRequest(BaseModel):
    algCode: str
    startTime: Optional[str] = ""
    endTime: Optional[str] = ""
    interval: Optional[int] = 60
    command: int = 1
    videoInfo: List[VideoInfo]
    rule: Optional[Rule] = None


class VideoTaskResultItem(BaseModel):
    analyseId: str
    devCode: str = ""
    osdVideoUrl: Optional[str] = ""


class VideoTaskResponse(BaseModel):
    resultCode: str = "200"
    resultValue: Optional[List[VideoTaskResultItem]] = None
    resultHint: Optional[str] = None


# ==================== 4.1.7.3 任务控制 ====================

class ControlTaskRequest(BaseModel):
    analyseId: str
    command: int


class ControlTaskResponse(BaseModel):
    resultCode: str = "200"
    resultValue: Optional[Any] = None
    resultHint: Optional[str] = None


# ==================== 4.1.7.4 视频分析结果 ====================

class ResultItem(BaseModel):
    score: Optional[float] = 0.0
    leftTopX: Optional[int] = 0
    leftTopY: Optional[int] = 0
    rightBottomX: Optional[int] = 0
    rightBottomY: Optional[int] = 0


class ResultDetail(BaseModel):
    algCode: Optional[str] = ""
    resultDesc: Optional[str] = ""
    num: Optional[int] = 0
    resultItems: Optional[List[ResultItem]] = []


class AnalyseResultRequest(BaseModel):
    algCode: Optional[str] = ""
    analyseId: str
    analyseTime: Optional[str] = ""
    analyseResults: List[str] = []
    rawImageName: Optional[str] = ""
    rawImageData: Optional[str] = ""
    osdImageName: Optional[str] = ""
    osdImageData: str = ""
    resultDetail: Optional[List[ResultDetail]] = []


class AnalyseResultResponse(BaseModel):
    resultCode: str = "200"
    resultHint: Optional[str] = ""


# ==================== 4.1.7.5 图片分析任务 ====================

class ImageTaskRequest(BaseModel):
    analyseId: str
    algCode: str
    imageData: str  # Base64编码图片
    rule: Optional[Rule] = None


class ImageTaskResultValue(BaseModel):
    analyseResults: List[str] = []
    analyseTime: Optional[str] = ""
    rawImageName: Optional[str] = ""
    rawImageData: Optional[str] = ""
    osdImageName: Optional[str] = ""
    osdImageData: Optional[str] = ""
    resultDetail: Optional[List[ResultDetail]] = []


class ImageTaskResponse(BaseModel):
    resultCode: str = "200"
    resultValue: Optional[ImageTaskResultValue] = None
    resultHint: Optional[str] = None


# ==================== 4.1.7.6 样本上传 ====================

class UploadSamplesRequest(BaseModel):
    fileId: str
    md5: str


class UploadSamplesResponse(BaseModel):
    resultCode: str = "200"
    resultValue: Optional[Dict[str, str]] = None
    resultHint: Optional[str] = None


# ==================== 4.1.7.7 保活 ====================

class KeepAliveRequest(BaseModel):
    devIP: str
    devPort: int


class KeepAliveResponse(BaseModel):
    resultCode: str = "200"
    resultValue: Optional[Dict[str, str]] = None
    resultHint: Optional[str] = None


# ==================== 4.1.7.8 更新分析ID ====================

class UpdateAnalyseIDRequest(BaseModel):
    oldAnalyseId: str
    newAnalyseId: str


class UpdateAnalyseIDResponse(BaseModel):
    resultCode: str = "200"
    resultValue: Optional[Dict[str, str]] = None
    resultHint: Optional[str] = None
