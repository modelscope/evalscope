from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_serializer, model_validator
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class TargetConfig(FrozenModel):
    model: str
    protocol: str = 'openai_chat'
    kind: Literal['remote', 'local_transformers', 'local_vllm'] = 'remote'
    base_url: str = 'http://127.0.0.1:8877/v1'
    api_key: Optional[str] = Field(default=None, exclude=True)
    headers: Dict[str, str] = Field(default_factory=dict)
    tokenizer: Optional[str] = None
    port: PositiveInt = 8877
    connect_timeout: Optional[PositiveFloat] = None
    read_timeout: Optional[PositiveFloat] = None
    total_timeout: Optional[PositiveFloat] = 21600
    skip_connection_test: bool = False
    attn_implementation: Optional[str] = None

    @field_serializer('headers')
    def redact_sensitive_headers(self, value: Dict[str, str]) -> Dict[str, str]:
        sensitive = {'authorization', 'proxy-authorization', 'x-api-key', 'api-key'}
        return {key: ('***' if key.lower() in sensitive else item) for key, item in value.items()}


class WorkloadConfig(FrozenModel):
    name: str = 'openqa'
    path: Optional[str] = None
    data_source: Literal['modelscope', 'huggingface', 'local'] = 'modelscope'
    prompt: Optional[str] = None
    min_prompt_length: int = Field(default=0, ge=0)
    max_prompt_length: PositiveInt = 131072
    options: Dict[str, Any] = Field(default_factory=dict)


class GenerationConfig(FrozenModel):
    max_tokens: Union[int, tuple[int, int]] = 2048
    min_tokens: Optional[int] = Field(default=None, ge=0)
    temperature: float = Field(default=0.0, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    logprobs: Optional[bool] = None
    n_choices: Optional[PositiveInt] = None
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    stream: bool = True
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_max_tokens(self) -> 'GenerationConfig':
        if isinstance(self.max_tokens, tuple):
            if len(self.max_tokens) != 2 or self.max_tokens[0] < 0 or self.max_tokens[0] > self.max_tokens[1]:
                raise ValueError('max_tokens range must be a non-negative (min, max) tuple')
        elif self.max_tokens < 0:
            raise ValueError('max_tokens must be non-negative')
        return self


class WarmupConfig(FrozenModel):
    count: Optional[int] = Field(default=None, ge=0)
    ratio: Optional[float] = Field(default=None, gt=0, lt=1)

    @model_validator(mode='after')
    def validate_choice(self) -> 'WarmupConfig':
        if self.count is not None and self.ratio is not None:
            raise ValueError('warmup count and ratio are mutually exclusive')
        return self

    def resolve(self, total: int) -> int:
        if self.count is not None:
            return self.count
        if self.ratio is not None:
            return max(1, int(total * self.ratio))
        return 0


class BaseLoad(FrozenModel):
    duration: Optional[PositiveFloat] = None
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)


class ClosedLoopLoad(BaseLoad):
    mode: Literal['closed_loop'] = 'closed_loop'
    concurrency: PositiveInt = 1
    request_count: Optional[PositiveInt] = None

    @model_validator(mode='after')
    def validate_limit(self) -> 'ClosedLoopLoad':
        if self.request_count is None and self.duration is None:
            raise ValueError('request_count or duration is required')
        return self


class OpenLoopLoad(BaseLoad):
    mode: Literal['open_loop'] = 'open_loop'
    request_rate: PositiveFloat
    request_count: Optional[PositiveInt] = None
    max_outstanding: PositiveInt = 512
    overflow_policy: Literal['record_drop', 'fail'] = 'record_drop'
    arrival: Literal['poisson', 'calibrated_poisson', 'constant'] = 'poisson'

    @model_validator(mode='after')
    def validate_limit(self) -> 'OpenLoopLoad':
        if self.request_count is None and self.duration is None:
            raise ValueError('request_count or duration is required')
        return self


class ConversationLoad(BaseLoad):
    mode: Literal['conversation'] = 'conversation'
    concurrency: PositiveInt = 1
    conversation_count: Optional[PositiveInt] = None
    max_turns: Optional[PositiveInt] = None

    @model_validator(mode='after')
    def validate_limit(self) -> 'ConversationLoad':
        if self.conversation_count is None and self.duration is None:
            raise ValueError('conversation_count or duration is required')
        return self


LoadSpec = Annotated[Union[ClosedLoopLoad, OpenLoopLoad, ConversationLoad], Field(discriminator='mode')]


class BenchmarkSuite(FrozenModel):
    loads: List[LoadSpec]
    sleep_between_runs: float = Field(default=0.0, ge=0)

    @model_validator(mode='after')
    def validate_loads(self) -> 'BenchmarkSuite':
        if not self.loads:
            raise ValueError('at least one load specification is required')
        return self


class RuntimeConfig(FrozenModel):
    seed: Optional[int] = None
    dataset_workers: int = Field(default=0, ge=0)
    queue_size: PositiveInt = 1024
    db_commit_interval: PositiveInt = 1000
    log_every: PositiveInt = 100
    debug: bool = False
    progress: bool = False
    visualizer: Optional[Literal['wandb', 'swanlab', 'clearml']] = None
    visualizer_project: str = 'evalscope-perf'
    visualizer_name: Optional[str] = None


class MetricsConfig(FrozenModel):
    last_window_seconds: PositiveFloat = 10.0
    steady_state_warmup_ratio: float = Field(default=0.1, ge=0, lt=1)


class OutputConfig(FrozenModel):
    root: str = 'outputs/perf'
    run_id: Optional[str] = None
    overwrite: bool = False
    html_report: bool = True
    console_report: bool = True

    @model_validator(mode='after')
    def validate_run_id(self) -> 'OutputConfig':
        if self.run_id is not None:
            if self.run_id in {'.', '..'} or '/' in self.run_id or '\\' in self.run_id or '\x00' in self.run_id:
                raise ValueError('run_id must be a single safe path component')
        return self


class SLAConfig(FrozenModel):
    variable: Literal['concurrency', 'request_rate']
    criteria: List[Dict[str, str]] = Field(default_factory=list)
    objective: Optional[Literal['max_rps', 'max_output_tps', 'min_latency']] = None
    lower_bound: PositiveInt = 1
    upper_bound: PositiveInt = 1024
    repetitions: PositiveInt = 3
    pass_ratio: float = Field(default=1.0, gt=0, le=1)

    @model_validator(mode='after')
    def validate_search(self) -> 'SLAConfig':
        if bool(self.criteria) == bool(self.objective):
            raise ValueError('SLA config requires exactly one of criteria or objective')
        if self.lower_bound > self.upper_bound:
            raise ValueError('SLA lower_bound must be <= upper_bound')
        return self


class PerfConfig(FrozenModel):
    target: TargetConfig
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    suite: BenchmarkSuite
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    sla: Optional[SLAConfig] = None

    @classmethod
    def from_input(cls, value: Union['PerfConfig', Mapping[str, Any]]) -> 'PerfConfig':
        return value if isinstance(value, cls) else cls.model_validate(value)
