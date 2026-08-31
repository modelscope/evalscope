from typing import Any, Dict, List, Literal, Optional

from .common import ApiResponseModel


class EvalInvokeResponse(ApiResponseModel):
    status: Literal['ok', 'completed', 'error', 'stopped']
    task_id: str
    result: Any = None
    table: Optional[str] = None
    error: Optional[str] = None


class BenchmarkDescriptionLocale(ApiResponseModel):
    full: str
    sections: Dict[str, str]


class BenchmarkDescription(ApiResponseModel):
    en: Optional[BenchmarkDescriptionLocale] = None
    zh: Optional[BenchmarkDescriptionLocale] = None


class BenchmarkEntry(ApiResponseModel):
    name: str
    pretty_name: str
    tags: List[str]
    category: Literal['llm', 'vlm', 'agent', 'aigc']
    subset_list: List[str]
    total_samples: int
    few_shot_num: int
    dataset_id: str
    paper_url: Optional[str] = None
    metrics: List[str]
    meta: Dict[str, Any]
    description: BenchmarkDescription


class BenchmarksResponse(ApiResponseModel):
    text: Optional[List[BenchmarkEntry]] = None
    multimodal: Optional[List[BenchmarkEntry]] = None
    agent: Optional[List[BenchmarkEntry]] = None
    aigc: Optional[List[BenchmarkEntry]] = None
