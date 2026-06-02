# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import copy
import json
import os
from argparse import Namespace
from pydantic import Field, field_validator, model_validator
from typing import Annotated, Any, Dict, List, Optional, Union

from evalscope.agent.external.config import ExternalAgentConfig
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.model import GenerateConfig, Model, ModelAPI
from evalscope.constants import (
    DEFAULT_DATASET_CACHE_DIR,
    DEFAULT_WORK_DIR,
    EvalBackend,
    EvalType,
    HubType,
    JudgeStrategy,
    ModelTask,
)
from evalscope.utils.argument_utils import BaseArgument, parse_int_or_float
from evalscope.utils.deprecation_utils import deprecated_warning
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import dict_to_yaml, gen_hash, json_to_dict, safe_filename, yaml_to_dict
from evalscope.utils.logger import get_logger
from evalscope.version import __version__ as _evalscope_version

AgentConfigUnion = Annotated[
    Union[NativeAgentConfig, ExternalAgentConfig],
    Field(discriminator='mode'),
]

logger = get_logger()

# Default configurations
DEFAULT_IMAGE_GEN_CONFIG = {
    'height': 1024,
    'width': 1024,
    'num_inference_steps': 50,
    'guidance_scale': 9.0,
}

DEFAULT_TEXT_GEN_CHECKPOINT_CONFIG = {
    'max_tokens': 2048,
    'do_sample': False,
    'top_k': 50,
    'top_p': 1.0,
    'temperature': 1.0,
    'n': 1,
}

DEFAULT_TEXT_GEN_SERVICE_CONFIG = {
    'temperature': 0.0,
}

DEFAULT_MODEL_ARGS_CHECKPOINT = {
    'revision': 'master',
    'precision': 'torch.float16',
}


class SandboxTaskConfig(BaseArgument):
    """Unified sandbox configuration for both pooled (SandboxMixin) and
    per-sample (EnclaveAgentEnvironment) execution paths.

    This is the forward-looking replacement for the legacy top-level
    ``TaskConfig.use_sandbox`` / ``sandbox_type`` / ``sandbox_manager_config``
    triplet.  :meth:`TaskConfig._init_default_sandbox_config` folds those
    legacy fields into ``self.sandbox`` once at construction time; afterwards
    all consumers read from ``self.sandbox`` and the legacy fields are
    untouched aliases retained solely for input compatibility.
    """

    enabled: bool = False
    """Whether to enable the sandbox subsystem for this task."""

    engine: str = 'docker'
    """Sandbox engine name.  One of ``'docker'`` / ``'volcengine'`` (or aliases
    accepted by :func:`evalscope.api.sandbox.resolve_engine`)."""

    default_config: Dict[str, Any] = Field(default_factory=dict)
    """Task-level overrides merged on top of ``BenchmarkMeta.sandbox_config``.
    The merged dict is passed to :func:`build_sandbox_config`.  Also acts as
    the default sandbox config for per-sample agent environments."""

    manager_config: Dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the ms_enclave manager constructor (e.g.
    ``base_url`` for a remote docker daemon, or volcengine credentials)."""

    pool_size: Optional[int] = None
    """Warm-pool size for pooled execution.  Defaults to ``eval_batch_size``
    when ``None``."""


class TaskConfig(BaseArgument):
    # Model-related arguments
    model: Optional[Union[str, Model, ModelAPI]] = None
    """The model to be evaluated. Can be a string path, Model object, or ModelAPI object."""

    model_id: Optional[str] = None
    """Unique identifier for the model. Auto-generated from model name if not provided."""

    model_args: Dict = Field(default_factory=dict)
    """Additional arguments to pass to the model during initialization."""

    model_task: str = ModelTask.TEXT_GENERATION
    """The type of task the model performs (e.g., text generation, image generation)."""

    # Template-related arguments
    chat_template: Optional[str] = None
    """Chat template to use for formatting conversations with the model."""

    # Dataset-related arguments
    datasets: List[str] = Field(default_factory=list)
    """List of dataset names to evaluate the model on."""

    dataset_args: Dict = Field(default_factory=dict)
    """Additional arguments to pass to datasets during loading."""

    dataset_dir: str = DEFAULT_DATASET_CACHE_DIR
    """Directory where datasets are cached locally."""

    dataset_hub: str = HubType.MODELSCOPE
    """Hub platform to download datasets from (e.g., ModelScope, HuggingFace)."""

    repeats: int = 1
    """Number of times to repeat the dataset items for k-metrics evaluation."""

    # Generation configuration arguments
    generation_config: Union[Dict, GenerateConfig] = Field(default_factory=dict)
    """Configuration parameters for text/image generation."""

    # Evaluation-related arguments
    eval_type: Optional[str] = None
    """Evaluation backend type. One of: 'llm_ckpt' (local checkpoint), 'openai_api',
    'openai_responses_api', 'anthropic_api', 'litellm', 'mock_llm', 'text2image', 'text2speech',
    'image_editing', 'custom'. Deprecated aliases: 'checkpoint' -> 'llm_ckpt',
    'server' -> 'openai_api'."""

    eval_backend: str = EvalBackend.NATIVE
    """Backend framework to use for evaluation."""

    eval_config: Union[str, Dict, None] = None
    """Additional evaluation configuration parameters."""

    limit: Optional[Union[int, float]] = None
    """Maximum number of samples to evaluate. Can be int (count) or float (fraction)."""

    eval_batch_size: int = 1
    """Batch size / concurrency for evaluation, applied across all stages:
    - Inference: concurrent requests (service mode) or batch size (checkpoint mode).
    - LLM-judge review (BatchReviewer Pass 1): number of concurrent threads.
    - batch_calculate_metrics (BatchReviewer Pass 2): number of samples per batch window.
    - Sandbox execution: worker pool size.
    """

    # Cache and working directory arguments
    use_cache: Optional[str] = None
    """Path to a previous output directory (e.g. 'outputs/20260519_120000') to resume from.
    Reuses cached predictions and reviews matched by sample_id; set None to start fresh."""

    rerun_review: bool = False
    """When use_cache is set, force re-running the review/scoring step
    (deletes existing reviews cache) while still reusing prediction cache."""

    work_dir: str = DEFAULT_WORK_DIR
    """Root directory for evaluation outputs (predictions/, reviews/, reports/, logs/).
    A timestamped subdirectory is appended unless `no_timestamp=True` or `use_cache` is set."""

    no_timestamp: bool = False
    """Do not add timestamp to the work_dir to avoid overwriting previous results."""

    enable_progress_tracker: bool = False
    """Whether to write a progress.json file tracking hierarchical evaluation progress.
    When True, each TqdmLogging instance auto-reports its stage to the file-backed
    ProgressTracker so the service layer can expose a real-time /progress endpoint."""

    # Debug and runtime mode arguments
    ignore_errors: bool = False
    """Whether to continue evaluation when encountering errors."""

    debug: bool = False
    """Enable debug mode for detailed logging and error reporting."""

    seed: Optional[int] = 42
    """Random seed for reproducible results."""

    api_url: Optional[str] = None
    """API endpoint URL for server-based model evaluation."""

    api_key: Optional[str] = 'EMPTY'
    """API key for authenticating with server-based models."""

    timeout: Optional[float] = None
    """[Deprecated] Use `generation_config.timeout` instead. Will be removed in v2.0.0.
    When set, value is forwarded to `generation_config.timeout` with a deprecation warning."""

    stream: Optional[bool] = None
    """[Deprecated] Use `generation_config.stream` instead. Will be removed in v2.0.0.
    When set, value is forwarded to `generation_config.stream` with a deprecation warning."""

    # LLMJudge arguments
    judge_strategy: str = JudgeStrategy.AUTO
    """How to score model outputs. One of:
    - 'auto': follow the benchmark's default (LLM judge only if the benchmark needs one).
    - 'rule': force rule-based scoring; never invoke an LLM judge.
    - 'llm': force LLM judge for every sample.
    - 'llm_recall': run rule-based scoring first, then pass the rule score to the LLM
       judge to produce the final score (useful when LLM should refine/recall rule misses)."""

    judge_worker_num: Optional[int] = None
    """[Deprecated] Use `eval_batch_size` instead. Will be removed in v2.0.0."""

    judge_model_args: Optional[Dict] = Field(default_factory=dict)
    """Additional arguments for the judge model configuration."""

    analysis_report: bool = False
    """Whether to generate detailed analysis reports after evaluation."""

    collect_perf: bool = True
    """Whether to collect per-request performance metrics (latency, TTFT, token usage)
    during evaluation. TTFT requires streaming (set `generation_config.stream=True`)."""

    # Sandbox configuration arguments
    sandbox: Optional[SandboxTaskConfig] = None
    """Unified sandbox configuration (preferred).  When set, takes precedence
    over the legacy ``use_sandbox`` / ``sandbox_type`` / ``sandbox_manager_config``
    fields which are kept as deprecated aliases."""

    use_sandbox: bool = False
    """[Deprecated] Use ``sandbox.enabled`` instead.  Kept as an alias for
    backward compatibility; will be removed in a future release."""

    sandbox_type: Optional[str] = 'docker'
    """[Deprecated] Use ``sandbox.engine`` instead.  Kept as an alias for
    backward compatibility; will be removed in a future release."""

    sandbox_manager_config: Optional[Dict] = Field(default_factory=dict)
    """[Deprecated] Use ``sandbox.manager_config`` instead.  Kept as an
    alias for backward compatibility; will be removed in a future release."""

    # Agent configuration (native AgentLoop OR external-agent bridge,
    # discriminated by the ``mode`` field on the embedded config).
    agent_config: Optional[AgentConfigUnion] = None
    """Per-task agent configuration.

    Discriminated union driven by the ``mode`` field:

    * ``mode='native'`` (default) → :class:`NativeAgentConfig`; every
      DefaultDataAdapter-based benchmark routes inference through the
      :class:`AgentLoop`.
    * ``mode='external'`` → :class:`ExternalAgentConfig`; inference is
      delegated to a third-party CLI (claude-code, mock, ...) and the
      bridge captures the LLM traffic into the same :class:`AgentTrace`.

    AgentAdapter subclasses (e.g. SWE-bench_Pro) ignore this field and use
    their own settings.  ``dict`` inputs accept ``{'mode': 'external',
    'framework': 'claude-code'}`` style payloads."""

    evalscope_version: Optional[str] = _evalscope_version
    """EvalScope version used for the evaluation."""

    # --- Field validators (single-field logic) ---

    @field_validator('limit', mode='before')
    @classmethod
    def _validate_limit(cls, v):
        if v is not None:
            v = parse_int_or_float(v)
            if v < 0:
                raise ValueError(f'`limit` must be >= 0 or None, got {v}.')
            if v == 0:
                return None
        return v

    @field_validator('eval_config', mode='before')
    @classmethod
    def _validate_eval_config(cls, v):
        if not v:
            return v
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            extension = os.path.splitext(v)[-1]
            if extension in ['.yaml', '.yml']:
                return yaml_to_dict(v)
            elif extension == '.json':
                return json_to_dict(v)
            else:
                try:
                    return json.loads(v)
                except Exception as e:
                    raise ValueError('eval_config string is not a valid json string or file path.') from e
        else:
            raise ValueError('eval_config should be a dict or a file path string.')

    @field_validator('agent_config', mode='before')
    @classmethod
    def _validate_agent_config(cls, v):
        if v is None or isinstance(v, (NativeAgentConfig, ExternalAgentConfig)):
            return v
        if isinstance(v, dict):
            mode = v.get('mode', 'native')
            if mode == 'external':
                return ExternalAgentConfig.model_validate(v)
            if mode == 'native':
                return NativeAgentConfig.model_validate(v)
            raise ValueError(f'`agent_config.mode` must be "native" or "external", got {mode!r}.')
        raise ValueError(
            f'`agent_config` must be a dict, NativeAgentConfig, ExternalAgentConfig or None, '
            f'got {type(v).__name__}.'
        )

    @field_validator('sandbox', mode='before')
    @classmethod
    def _validate_sandbox(cls, v):
        if v is None or isinstance(v, SandboxTaskConfig):
            return v
        if isinstance(v, dict):
            return SandboxTaskConfig.model_validate(v)
        raise ValueError(f'`sandbox` must be a dict, SandboxTaskConfig or None, got {type(v).__name__}.')

    # --- Model validator (cross-field logic, replaces __post_init__) ---

    @model_validator(mode='after')
    def _post_init(self) -> 'TaskConfig':
        self._init_model_and_id()
        self._init_default_generation_config()
        self._init_default_model_args()
        self._init_default_sandbox_config()
        self._parse_rag_eval_config()

        # Handle deprecated judge_worker_num -> eval_batch_size
        if self.judge_worker_num is not None:
            deprecated_warning(
                logger, 'The `judge_worker_num` parameter is deprecated and will be removed in v2.0.0. '
                'Use `eval_batch_size` instead.'
            )

        return self

    def _parse_rag_eval_config(self):
        """Parse eval_config into typed Pydantic models for RAGEval backend."""
        if self.eval_backend != EvalBackend.RAG_EVAL or not isinstance(self.eval_config, dict):
            return
        tool = self.eval_config.get('tool', '').lower()
        if tool == 'mteb':
            from evalscope.backend.rag_eval.mteb.arguments import MTEBToolConfig
            self.eval_config = MTEBToolConfig(**self.eval_config)
        elif tool == 'ragas':
            from evalscope.backend.rag_eval.ragas.arguments import RAGASToolConfig
            self.eval_config = RAGASToolConfig(**self.eval_config)
        elif tool == 'clip_benchmark':
            from evalscope.backend.rag_eval.clip_benchmark.arguments import ClipBenchmarkToolConfig
            self.eval_config = ClipBenchmarkToolConfig(**self.eval_config)

    def _init_model_and_id(self):
        # Set model to DummyCustomModel if not provided
        if self.model is None:
            logger.info('No model is provided, using DummyCustomModel for testing.')
            self.model = self.model_task
            self.eval_type = EvalType.MOCK_LLM

        # Set eval_type to openai_api if api_url is provided
        if self.api_url is not None and self.eval_type is None:
            logger.info("api_url is provided, setting eval_type to 'openai_api'.")
            self.eval_type = EvalType.OPENAI_API

        # Set eval_type to CHECKPOINT if model is a string path and eval_type is not set
        if self.model and self.eval_type is None:
            logger.info('No eval_type is provided, setting eval_type to CHECKPOINT.')
            self.eval_type = EvalType.CHECKPOINT

        # Set model_id if not provided
        if not self.model_id:
            self.model_id = self._infer_model_id()

    def _infer_model_id(self) -> str:
        if isinstance(self.model, str):
            return safe_filename(os.path.basename(self.model))
        elif isinstance(self.model, Model):
            return safe_filename(self.model.name)
        elif isinstance(self.model, ModelAPI):
            return safe_filename(self.model.model_name)
        return 'dummy_model'

    def _init_default_generation_config(self):
        # 1. Set defaults if empty
        if not self.generation_config:
            self.generation_config = self._get_default_generation_config()

        # 2. Validate/Convert to GenerateConfig object
        if isinstance(self.generation_config, dict):
            self.generation_config = GenerateConfig.model_validate(self.generation_config)

        # 3. Sync batch size
        self.generation_config.batch_size = self.eval_batch_size

        # 4. Handle deprecations
        self._handle_generation_config_deprecations()

    def _get_default_generation_config(self) -> Dict:
        if self.model_task == ModelTask.IMAGE_GENERATION:
            return DEFAULT_IMAGE_GEN_CONFIG.copy()

        elif self.model_task == ModelTask.TEXT_GENERATION:
            if self.eval_type == EvalType.CHECKPOINT:
                return DEFAULT_TEXT_GEN_CHECKPOINT_CONFIG.copy()
            elif self.eval_type in (EvalType.OPENAI_API, EvalType.OPENAI_RESPONSES_API):
                return DEFAULT_TEXT_GEN_SERVICE_CONFIG.copy()

        return {}

    def _handle_generation_config_deprecations(self):
        assert isinstance(self.generation_config, GenerateConfig)

        if self.timeout is not None:
            deprecated_warning(
                logger,
                'The `timeout` parameter is deprecated and will be removed in v2.0.0. Use `generation_config.timeout` instead.'
            )
            self.generation_config.timeout = self.timeout

        if self.stream is not None:
            deprecated_warning(
                logger,
                'The `stream` parameter is deprecated and will be removed in v2.0.0. Use `generation_config.stream` instead.'
            )
            self.generation_config.stream = self.stream

        if self.generation_config.n is not None and self.generation_config.n > 1:
            self.repeats = self.generation_config.n
            self.generation_config.n = 1
            deprecated_warning(
                logger,
                'The `n` parameter in generation_config is deprecated and will be removed in v2.0.0. Use `TaskConfig.repeats` instead.'
            )

    def _init_default_model_args(self):
        if self.model_args:
            return
        if self.model_task == ModelTask.TEXT_GENERATION and self.eval_type == EvalType.CHECKPOINT:
            self.model_args = DEFAULT_MODEL_ARGS_CHECKPOINT.copy()

    def _init_default_sandbox_config(self):
        """Normalise sandbox configuration into ``self.sandbox``.

        After this method every consumer (``SandboxMixin``, data adapters,
        ``EnclaveAgentEnvironment``) reads sandbox settings exclusively from
        ``self.sandbox`` — the legacy ``use_sandbox`` / ``sandbox_type`` /
        ``sandbox_manager_config`` fields are single-source-of-truth inputs
        only and are **not** kept in sync afterwards.

        Rules:
          * If ``sandbox`` is provided, it wins; when legacy fields are also
            set a deprecation warning is emitted.
          * Otherwise ``sandbox`` is constructed from the legacy fields so
            historical task configs keep working.
        """
        legacy_set = bool(self.use_sandbox) or (self.sandbox_type
                                                not in (None, 'docker')) or bool(self.sandbox_manager_config)

        if self.sandbox is None:
            # Build from legacy fields (possibly all defaults).
            self.sandbox = SandboxTaskConfig(
                enabled=bool(self.use_sandbox),
                engine=self.sandbox_type or 'docker',
                manager_config=dict(self.sandbox_manager_config or {}),
            )
        elif legacy_set:
            deprecated_warning(
                logger, 'Both `sandbox` and legacy sandbox fields '
                '(`use_sandbox` / `sandbox_type` / `sandbox_manager_config`) are set; '
                'the nested `sandbox` object takes precedence. The legacy fields will be '
                'removed in a future release.'
            )

        if not self.sandbox.enabled:
            return

        check_import('ms_enclave', 'evalscope[sandbox]', raise_error=True)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base, returning a new dict."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = TaskConfig._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def update(self, other: Union['TaskConfig', dict]):
        if isinstance(other, TaskConfig):
            other = other.to_dict()
        merged = self._deep_merge(self.to_dict(), other)
        for key, value in merged.items():
            setattr(self, key, value)

    def dump_yaml(self, output_dir: str):
        """Dump the task configuration to a YAML file."""
        task_cfg_file = os.path.join(output_dir, f'task_config.yaml')
        try:
            logger.info(f'Dump task config to {task_cfg_file}')
            dict_to_yaml(self.to_dict(), task_cfg_file)
        except Exception as e:
            logger.warning(f'Failed to dump overall task config: {e}')

    def to_dict(self):
        result = self.model_dump()

        # Remove sensitive info
        result.pop('api_key', None)

        # Handle nested sensitive info in judge_model_args
        if self.judge_model_args:
            result['judge_model_args'] = copy.deepcopy(self.judge_model_args)
            result['judge_model_args'].pop('api_key', None)

        # Serialize Model objects
        if isinstance(self.model, (Model, ModelAPI)):
            result['model'] = self.model.__class__.__name__

        # Serialize GenerateConfig
        if isinstance(self.generation_config, GenerateConfig):
            result['generation_config'] = self.generation_config.model_dump(exclude_unset=True)

        return result


def parse_task_config(task_cfg) -> TaskConfig:
    """Parse task configuration from various formats into a TaskConfig object."""
    if isinstance(task_cfg, TaskConfig):
        logger.info('Args: Task config is provided with TaskConfig type.')
    elif isinstance(task_cfg, dict):
        logger.info('Args: Task config is provided with dictionary type.')
        task_cfg = TaskConfig.from_dict(task_cfg)
    elif isinstance(task_cfg, Namespace):
        logger.info('Args: Task config is provided with CommandLine type.')
        task_cfg = TaskConfig.from_args(task_cfg)
    elif isinstance(task_cfg, str):
        extension = os.path.splitext(task_cfg)[-1]
        logger.info(f'Args: Task config is provided with {extension} file type.')
        if extension in ['.yaml', '.yml']:
            task_cfg = TaskConfig.from_yaml(task_cfg)
        elif extension == '.json':
            task_cfg = TaskConfig.from_json(task_cfg)
        else:
            raise ValueError('Args: Unsupported file extension.')
    else:
        raise ValueError('Args: Please provide a valid task config.')
    return task_cfg
