import argparse
import json
import os
from contextlib import contextmanager
from pydantic import Field, field_validator, model_validator
from typing import Any, Dict, List, Optional, Union

from evalscope.constants import DEFAULT_WORK_DIR, VisualizerType
from evalscope.perf.multi_turn_args import IntOrRange, MultiTurnArgs, _sample_int_or_range
from evalscope.utils import BaseArgument
from evalscope.utils.logger import get_logger

logger = get_logger()

_OPENAI_API_ENDPOINT_MAP = {
    'openai': 'chat/completions',
    'openai_responses': 'responses',
    'openai_response': 'responses',
    'responses': 'responses',
    'openai_embedding': 'embeddings',
    'embedding': 'embeddings',
    'openai_rerank': 'reranks',
    'rerank': 'reranks',
}


class Arguments(BaseArgument):
    # Model and API
    model: str
    """Model name or path."""

    model_id: Optional[str] = None
    """Model identifier."""

    attn_implementation: Optional[str] = None
    """Attention implementation, only for local inference."""

    api: str = 'openai'
    """Service API protocol. One of: 'openai' (chat/completions), 'openai_responses'
    (Responses API), 'openai_embedding'/'embedding' (embeddings), 'openai_rerank'/'rerank'
    (reranks), or 'local'/'local_vllm' for in-process inference. The endpoint path is
    auto-appended onto `--url` based on this value."""

    tokenizer_path: Optional[str] = None
    """Path to the tokenizer."""

    port: int = 8877
    """Port number for the local API server."""

    # Connection settings
    url: str = 'http://127.0.0.1:8877/v1/chat/completions'
    """URL for the API connection."""

    headers: Dict[str, Any] = Field(default_factory=dict)
    """Custom headers."""

    connect_timeout: Optional[int] = None
    """Connection timeout in seconds."""

    read_timeout: Optional[int] = None
    """Read timeout in seconds."""

    total_timeout: Optional[int] = 6 * 60 * 60
    """Total timeout in seconds."""

    api_key: Optional[str] = None
    """The API key for authentication."""

    no_test_connection: bool = False
    """Skip the connection test before starting the benchmark."""

    # Performance and parallelism
    number: Union[int, List[int]] = 1000
    """Number of requests to be made."""

    parallel: Union[int, List[int]] = 1
    """Number of parallel requests."""

    rate: Union[float, List[float]] = -1
    """Rate limit for requests per second (default: -1, no limit). Supports a list of values for multi-run sweeps in open-loop mode."""

    warmup_num: Union[int, float] = 0
    """Number or ratio of warmup requests.

    - 0: disabled (default).
    - >= 1 (int or float): absolute number of warmup requests.
    - 0 < value < 1 (float): ratio of ``--number``, e.g. 0.1 = 10% warmup.
      Actual count = max(1, int(warmup_num * number)). Useful for sweep mode
      where each run has a different number of requests.
    """

    @property
    def warmup_count(self) -> int:
        """Resolved number of warmup requests/conversations.

        Derived from ``warmup_num`` and the current ``number`` value.
        Handles both absolute count (>=1) and ratio (0 < value < 1) modes.
        """
        if self.warmup_num <= 0:
            return 0
        n = self.number if isinstance(self.number, int) else self.number[0]
        if self.warmup_num >= 1:
            return int(self.warmup_num)
        return max(1, int(self.warmup_num * n))

    @property
    def total_count(self) -> int:
        """Total number of requests/conversations including warmup.

        Equal to ``number + warmup_count``.  When ``number`` is a list
        (sweep mode), uses the first element.
        """
        n = self.number if isinstance(self.number, int) else self.number[0]
        return n + self.warmup_count

    open_loop: bool = False
    """Enable open-loop rate mode: dispatch requests at the scheduled rate without semaphore backpressure.

    When enabled, requests are fired according to the Poisson arrival schedule set by ``--rate``
    regardless of whether the server has finished processing previous requests.
    Use ``--rate`` (list) and ``--number`` (list) to sweep multiple load levels.

    Semantics in open-loop mode:
    - ``--rate``: target request rate in req/s; drives multi-run iteration (replaces ``--parallel`` sweep).
    - ``--number``: total requests per run; must have the same length as ``--rate``.
    - ``--parallel``: ignored for concurrency control; optionally used for MetricsAccumulator labelling.
    """

    sleep_interval: int = 5
    """Sleep interval between performance runs, in seconds."""

    duration: Optional[float] = None
    """Wall-clock budget for one benchmark run, in seconds.

    Soft-exit semantics (matches [applied-compute/trie](https://github.com/applied-compute/trie)):
    once the deadline is reached the dispatcher stops scheduling new requests,
    but already in-flight requests are awaited to completion.  In multi-turn
    mode "in-flight" is per-trace: an already-claimed conversation runs every
    remaining turn so per-trace metrics stay coherent and ``trace_summary.json``
    never contains partial traces.

    Honored in all benchmark modes (single-turn closed-loop, single-turn
    open-loop, and multi-turn).  Warmup phases ignore ``--duration`` and run
    in full; the deadline only applies to the benchmark phase.  When both
    ``--number`` and ``--duration`` are set, whichever limit is reached first
    ends the run.  Sweep mode (list-valued ``--number`` / ``--parallel`` /
    ``--rate``) applies ``--duration`` to each individual run.
    """

    # SLA Auto-tuning
    sla_auto_tune: bool = False
    """Enable SLA auto-tuning."""

    sla_variable: str = 'parallel'
    """Variable to tune: 'parallel' or 'rate'."""

    sla_params: Optional[List[Dict[str, Any]]] = None
    """SLA constraints in JSON format."""

    sla_num_runs: int = 3
    """Number of runs to average for each configuration in SLA auto-tuning."""

    sla_upper_bound: int = 65536
    """Upper bound limit for SLA auto-tuning."""

    sla_lower_bound: int = 1
    """Lower bound limit for SLA auto-tuning."""

    sla_fixed_parallel: Optional[int] = None
    """Fixed parallel workers used when tuning `rate`.

    If not set, falls back to `sla_upper_bound` for backward compatibility.
    """

    sla_number_multiplier: Optional[float] = None
    """Multiplier for number of requests relative to the tuned SLA variable (parallel/rate) in SLA auto-tuning.
    If set, number = round(val * sla_number_multiplier) where val is the current value of `sla_variable`
    (either `parallel` or `rate`). If None, defaults to 2 (i.e. number = val * 2)."""

    # Tuning knobs
    db_commit_interval: int = 1000
    """Number of rows buffered before committing to the DB."""

    queue_size_multiplier: int = 5
    """Maxsize for queue = parallel * this multiplier."""

    in_flight_task_multiplier: int = 2
    """Max scheduled tasks = parallel * this multiplier."""

    # Logging and debugging
    log_every_n_query: int = 100
    """Log every N queries."""

    debug: bool = False
    """Debug mode."""

    enable_progress_tracker: bool = False
    """Whether to write a progress.json file tracking hierarchical benchmark progress."""

    visualizer: Optional[str] = None
    """Visualizer for logging, supports 'swanlab' or 'wandb'."""

    wandb_api_key: Optional[str] = None
    """API key for wandb visualizer. [Deprecated] Prefer the WANDB_API_KEY env var; this field
    will be removed in a future release."""

    swanlab_api_key: Optional[str] = None
    """API key for swanlab visualizer. [Deprecated] Prefer the SWANLAB_API_KEY env var; this
    field will be removed in a future release."""

    swanlab_host: Optional[str] = None
    """SwanLab host URL for self-hosted deployments. Falls back to SWANLAB_HOST env var."""

    name: Optional[str] = None
    """Name for the run."""

    # Output settings
    outputs_dir: str = DEFAULT_WORK_DIR
    """Output directory."""

    no_timestamp: bool = False
    """Whether to disable timestamp in output directory."""

    # Prompt settings
    max_prompt_length: int = 131072
    """Maximum length of the prompt (in tokens if --tokenizer-path is set, otherwise in characters)."""

    min_prompt_length: int = 0
    """Minimum length of the prompt."""

    prefix_length: int = 0
    """Length of the prefix, only for random dataset."""

    prompt: Optional[str] = None
    """The prompt text."""

    query_template: Optional[str] = None
    """Template for the query."""

    apply_chat_template: Optional[bool] = None
    """Whether to apply chat template."""

    # random vl settings
    image_width: int = 224
    """Width of the image for random VL dataset."""

    image_height: int = 224
    """Height of the image for random VL dataset."""

    image_format: str = 'RGB'
    """Image format for random VL dataset."""

    image_num: int = 1
    """Number of images for random VL dataset."""

    image_patch_size: int = 28
    """Patch size for image tokenizer, only for local image token calculation."""

    # Dataset settings
    dataset: str = 'openqa'
    """Dataset name registered in evalscope.perf.plugin.datasets (default: 'openqa').
    Examples: 'openqa', 'random', 'random_vl', 'random_multi_turn', 'share_gpt_zh_multi_turn',
    'share_gpt_en_multi_turn', 'custom', 'custom_multi_turn', 'swe_smith', 'speed_benchmark*'."""

    dataset_path: Optional[str] = None
    """Path to the dataset."""

    dataset_offset: int = 0
    """Global token-sequence offset for random datasets.

    When running multiple parallel/rate sweeps, this offset is automatically
    incremented by the number of requests in each run so that successive runs
    start at different positions in the token vocabulary, preventing KV-cache
    hits from identical prompts.

    Set to 0 (default) to enable automatic cumulative offsetting.
    A non-zero initial value shifts the starting position of the first run.
    """

    # Response settings
    frequency_penalty: Optional[float] = None
    """Frequency penalty for the response."""

    repetition_penalty: Optional[float] = None
    """Repetition penalty for the response."""

    logprobs: Optional[bool] = None
    """Whether to log probabilities."""

    max_tokens: Optional[IntOrRange] = 2048
    """Maximum number of tokens in the response.

    Accepts an int or a ``[min, max]`` list for uniform sampling per request.
    """

    min_tokens: Optional[int] = None
    """Minimum number of tokens in the response."""

    n_choices: Optional[int] = None
    """Number of response choices."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    stop: Optional[List[str]] = None
    """Stop sequences for the response."""

    stop_token_ids: Optional[List[str]] = None
    """Stop token IDs for the response."""

    stream: Optional[bool] = True
    """Whether to stream the response."""

    temperature: float = 0.0
    """Temperature setting for the response."""

    top_p: Optional[float] = None
    """Top-p (nucleus) sampling setting for the response."""

    top_k: Optional[int] = None
    """Top-k sampling setting for the response."""

    extra_args: Optional[Dict[str, Any]] = None
    """Extra arguments."""

    tokenize_prompt: bool = False
    """Tokenize prompt client-side and send token IDs directly to /v1/completions,
    avoiding the token ID -> text -> token ID round-trip inflation.
    Requires --tokenizer-path to be set."""

    # Multi-turn settings
    multi_turn: bool = False
    """Enable multi-turn conversation benchmark mode.

    When enabled, each request is a single turn within a multi-turn conversation.
    The conversation context (previous turns + model responses) is accumulated and
    sent with each subsequent turn.  Use a multi-turn compatible dataset such as
    ``random_multi_turn``, ``share_gpt_zh_multi_turn``, or ``share_gpt_en_multi_turn``.

    Semantics of existing parameters in multi-turn mode:
    - ``--number``: total number of conversations to run.
    - ``--parallel``: number of conversations executed concurrently (each worker owns one conversation).
    """

    min_turns: int = 1
    """Minimum number of user turns per conversation.

    Used by ``random_multi_turn`` to set the lower bound of the per-conversation
    turn count.  For ``swe_smith`` live construction, each conversation samples
    its turn count uniformly from ``[min_turns, max_turns]``.
    """

    max_turns: Optional[int] = None
    """Maximum number of user turns per conversation.

    For ``random_multi_turn``: caps the randomly sampled turn count (required).
    For ``share_gpt_*_multi_turn`` / ``custom_multi_turn``: truncates long conversations.
    For ``swe_smith`` live construction: upper bound for per-conversation turn
    count sampling; defaults to ``min_turns`` when not set.
    """

    multi_turn_args: Optional[MultiTurnArgs] = None
    """Advanced multi-turn conversation parameters (MultiTurnArgs). Pass as JSON string via CLI."""

    # --- Field validators ---

    @field_validator('max_tokens', mode='before')
    @classmethod
    def _validate_max_tokens(cls, v):
        if isinstance(v, list):
            if len(v) == 1:
                return v[0]  # single value from nargs='+'
            if len(v) != 2:
                raise ValueError(f'--max-tokens accepts 1 or 2 values [min max], got {v}')
            if v[0] > v[1]:
                raise ValueError(f'--max-tokens range min must be <= max, got {v}')
            if v[0] < 0:
                raise ValueError(f'--max-tokens range values must be >= 0, got {v}')
        return v

    @field_validator('multi_turn_args', mode='before')
    @classmethod
    def _validate_multi_turn_args(cls, v):
        if v is None:
            return v
        if isinstance(v, MultiTurnArgs):
            return v
        if isinstance(v, str):
            try:
                return MultiTurnArgs(**json.loads(v))
            except Exception as e:
                raise ValueError(f'Failed to parse --multi-turn-args JSON: {e}') from e
        if isinstance(v, dict):
            return MultiTurnArgs(**v)
        return v

    @field_validator('rate', mode='before')
    @classmethod
    def _validate_rate(cls, v):
        if isinstance(v, (int, float)):
            return [float(v)]
        return v

    @field_validator('number', mode='before')
    @classmethod
    def _validate_number(cls, v):
        if isinstance(v, int):
            return [v]
        return v

    @field_validator('parallel', mode='before')
    @classmethod
    def _validate_parallel(cls, v):
        if isinstance(v, int):
            return [v]
        return v

    @field_validator('db_commit_interval', mode='after')
    @classmethod
    def _validate_db_commit_interval(cls, v):
        return max(v, 1) if v <= 0 else v

    @field_validator('queue_size_multiplier', mode='after')
    @classmethod
    def _validate_queue_size_multiplier(cls, v):
        return max(v, 1) if v <= 0 else v

    @field_validator('in_flight_task_multiplier', mode='after')
    @classmethod
    def _validate_in_flight_task_multiplier(cls, v):
        return max(v, 1) if v <= 0 else v

    # --- Model validator (cross-field logic) ---

    @model_validator(mode='after')
    def _post_init(self) -> 'Arguments':
        # Set the default headers
        self.headers = self.headers or {}
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'

        # Set the model ID based on the model name
        self.model_id = os.path.basename(self.model)

        # Set the URL based on the dataset type
        self._resolve_url()

        # Resolve apply_chat_template from the *original* URL before any redirects.
        if self.apply_chat_template is None:
            self.apply_chat_template = self.url.strip('/').endswith('chat/completions')

        # Validate sweep params (number/parallel/rate consistency)
        self._validate_sweep_params()

        return self

    def _resolve_url(self):
        """Resolve URL based on api, dataset, port, and tokenize_prompt settings."""
        # Set the URL based on the dataset type
        if self.api.startswith('local'):
            if self.dataset.startswith('speed_benchmark'):
                self.url = f'http://127.0.0.1:{self.port}/v1/completions'
            else:
                self.url = f'http://127.0.0.1:{self.port}/v1/chat/completions'

        # Auto-append endpoint path for openai-compatible APIs when URL has no known endpoint suffix
        if self.api in _OPENAI_API_ENDPOINT_MAP:
            _stripped = self.url.rstrip('/')
            _expected_suffix = _OPENAI_API_ENDPOINT_MAP[self.api]
            _known_endpoints = ('chat/completions', 'completions', 'responses', 'embeddings', 'reranks')
            if self.api in ('openai_responses', 'openai_response',
                            'responses') and _stripped.endswith('/chat/completions'):
                self.url = _stripped[:-len('chat/completions')] + 'responses'
                logger.warning(f'OpenAI Responses API selected: URL auto-adjusted to responses endpoint: {self.url}')
            elif not any(_stripped.endswith('/' + ep) for ep in _known_endpoints):
                self.url = _stripped + '/' + _expected_suffix
                logger.warning(
                    f'URL "{_stripped}" has no endpoint path, auto-appended "/{_expected_suffix}". '
                    'If this is not intended, please specify the full URL explicitly.'
                )

        # When tokenize_prompt is enabled, redirect to the completions endpoint.
        if self.tokenize_prompt:
            if self.api in ('openai_responses', 'openai_response', 'responses'):
                raise ValueError('--tokenize-prompt is not supported with the OpenAI Responses API.')
            if not self.tokenizer_path:
                raise ValueError('--tokenizer-path is required when --tokenize-prompt is set.')
            _stripped = self.url.rstrip('/')
            if _stripped.endswith('chat/completions'):
                self.url = _stripped[:-len('chat/completions')] + 'completions'
                logger.warning(
                    f'--tokenize-prompt is set: URL auto-adjusted from chat/completions '
                    f'to completions endpoint: {self.url}'
                )

    def _validate_sweep_params(self):
        """Validate number/parallel/rate consistency after normalization."""
        if self.open_loop:
            if len(self.number) != len(self.rate):
                raise ValueError(
                    f'In open-loop mode the length of --number and --rate must match, '
                    f'but got number: {self.number} and rate: {self.rate}'
                )
            if not all(r > 0 for r in self.rate):
                raise ValueError(f'In open-loop mode all --rate values must be > 0, but got: {self.rate}')
            # In open-loop mode concurrency is unbounded; set parallel=-1 so downstream
            # display layers render it as INF instead of a numeric value.
            self.parallel = [-1]
            logger.info(
                'open-loop mode enabled: concurrency is unbounded (parallel set to -1 / INF). '
                f'Rate sweep: {self.rate}, number sweep: {self.number}.'
            )
        else:
            if len(self.number) != len(self.parallel):
                raise ValueError(
                    f'The length of number and parallel should be the same, '
                    f'but got number: {self.number} and parallel: {self.parallel}'
                )

    @contextmanager
    def output_context(self, path: str):
        """
        Context manager for temporarily setting outputs_dir.

        Args:
            path: Path to set as outputs_dir

        Yields:
            The path that was set
        """
        original_path = self.outputs_dir
        try:
            self.outputs_dir = path
            yield path
        finally:
            self.outputs_dir = original_path


class ParseKVAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if not values:
            setattr(namespace, self.dest, {})
        else:
            try:
                kv_dict = {}
                for kv in values:
                    parts = kv.split('=', 1)  # only split the first '='
                    if len(parts) != 2:
                        raise ValueError(f'Invalid key-value pair: {kv}')
                    key, value = parts
                    kv_dict[key.strip()] = value.strip()
                setattr(namespace, self.dest, kv_dict)
            except ValueError as e:
                parser.error(f'Error parsing key-value pairs: {e}')


def add_argument(parser: argparse.ArgumentParser):
    # yapf: disable
    # Model and API
    parser.add_argument('--model', type=str, required=True, help='The test model name.')
    parser.add_argument('--attn-implementation', required=False, default=None, help='Attention implementaion')
    parser.add_argument('--api', type=str, default='openai', help='Service API protocol: openai | openai_responses/responses | openai_embedding/embedding | openai_rerank/rerank | local/local_vllm. Determines the auto-appended endpoint path on --url.')  # noqa: E501
    parser.add_argument(
        '--tokenizer-path', type=str, required=False, default=None, help='Specify the tokenizer weight path')

    # Connection settings
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8877/v1/chat/completions')
    parser.add_argument('--port', type=int, default=8877, help='The port for local inference')
    parser.add_argument('--headers', nargs='+', dest='headers', action=ParseKVAction, help='Extra HTTP headers')
    parser.add_argument('--api-key', type=str, required=False, default=None, help='The API key for authentication')
    parser.add_argument('--connect-timeout', type=int, required=False, default=None, help='The network connection timeout')  # noqa: E501
    parser.add_argument('--read-timeout', type=int, required=False, default=None, help='The network read timeout')
    parser.add_argument('--total-timeout', type=int, default=6 * 60 * 60, help='The total timeout for each request')  # noqa: E501
    parser.add_argument('--no-test-connection', action='store_true', default=False, help='Do not test the connection before starting the benchmark')  # noqa: E501

    # Performance and parallelism
    parser.add_argument('-n', '--number', type=int, default=1000, nargs='+', help='How many requests to be made')
    parser.add_argument('--parallel', type=int, default=1, nargs='+', help='Set number of concurrency requests, default 1')  # noqa: E501
    parser.add_argument('--rate', type=float, default=-1, nargs='+',
                        help='Number of requests per second. default -1 means no rate limit. '
                             'Accepts multiple values for open-loop multi-run sweeps, e.g. --rate 5 10 20')  # noqa: E501
    parser.add_argument('--warmup-num', type=float, default=0,
                        help='Number or ratio of warmup requests. '
                             '>=1: absolute count. 0<value<1: ratio of --number. '
                             '0: disabled. Default: 0')
    parser.add_argument('--open-loop', action='store_true', default=False,
                        help='Enable open-loop rate mode: dispatch requests at the scheduled rate without '
                             'semaphore backpressure. Use with --rate (list) and matching --number (list).')  # noqa: E501
    parser.add_argument(
        '--sleep-interval', type=int, default=5, help='Sleep interval between performance runs, in seconds. Default 5')  # noqa: E501
    parser.add_argument(
        '--duration', type=float, default=None,
        help='Wall-clock budget in seconds for one benchmark run (applies to all modes). '
             'Soft exit: once reached, no new requests are dispatched but in-flight ones run to '
             'completion (multi-turn finishes every remaining turn of any already-claimed trace, '
             'matching trie). Warmup ignores this. When both --number and --duration are set, '
             'whichever limit is reached first ends the run.')  # noqa: E501

    # SLA Auto-tuning
    parser.add_argument('--sla-auto-tune', action='store_true', default=False, help='Enable SLA auto-tuning')
    parser.add_argument('--sla-variable', type=str, default='parallel', choices=['parallel', 'rate'], help='The variable to tune, can be parallel or rate')  # noqa: E501
    parser.add_argument('--sla-params', type=json.loads, default=None, help='SLA constraints in JSON format')
    parser.add_argument('--sla-num-runs', type=int, default=3, help='Number of runs to average for each configuration in SLA auto-tuning')  # noqa: E501
    parser.add_argument('--sla-upper-bound', type=int, default=65536, help='Upper bound of the tuned SLA variable search range')  # noqa: E501
    parser.add_argument('--sla-lower-bound', type=int, default=1, help='Lower bound of the tuned SLA variable search range')  # noqa: E501
    parser.add_argument('--sla-fixed-parallel', type=int, default=None, help='Fixed parallel workers used when --sla-variable=rate. Defaults to --sla-upper-bound for backward compatibility.')  # noqa: E501
    parser.add_argument('--sla-number-multiplier', type=float, default=None, help='Multiplier for number of requests relative to parallel/rate in SLA auto-tuning. number = round(val * N). Defaults to 2 if not set.')  # noqa: E501

    # Tuning knobs
    parser.add_argument('--db-commit-interval', type=int, default=1000, help='Rows buffered before SQLite commit')
    parser.add_argument('--queue-size-multiplier', type=int, default=5, help='Queue maxsize = parallel * multiplier')
    parser.add_argument('--in-flight-task-multiplier', type=int, default=2, help='Max scheduled tasks = parallel * multiplier')  # noqa: E501

    # Logging and debugging
    parser.add_argument('--log-every-n-query', type=int, default=100, help='Logging every n query')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug request send')
    parser.add_argument('--visualizer', type=str, default=None,
                        choices=[VisualizerType.WANDB, VisualizerType.SWANLAB, VisualizerType.CLEARML, None], help='The visualizer to use, default None')  # noqa: E501
    parser.add_argument('--wandb-api-key', type=str, default=None, help='The wandb API key')
    parser.add_argument('--swanlab-api-key', type=str, default=None, help='The swanlab API key')
    parser.add_argument(
        '--swanlab-host',
        type=str,
        default=None,
        help='SwanLab host URL for self-hosted deployments (also reads SWANLAB_HOST env var)')
    parser.add_argument('--name', type=str, help='The wandb/swanlab/clearml result name and result db name')
    parser.add_argument('--enable-progress-tracker', action='store_true', default=False, help='Enable progress tracker')

    # Prompt settings
    parser.add_argument('--max-prompt-length', type=int, default=131072, help='Maximum input prompt length')
    parser.add_argument('--min-prompt-length', type=int, default=0, help='Minimum input prompt length')
    parser.add_argument('--prefix-length', type=int, default=0, help='The prefix length')
    parser.add_argument('--prompt', type=str, required=False, default=None, help='Specified the request prompt')
    parser.add_argument('--query-template', type=str, default=None, help='Specify the query template')
    parser.add_argument(
        '--apply-chat-template', type=argparse.BooleanOptionalAction, default=None, help='Apply chat template to the prompt')  # noqa: E501
    # random vl settings
    parser.add_argument('--image-width', type=int, default=224, help='Width of the image for random VL dataset')
    parser.add_argument('--image-height', type=int, default=224, help='Height of the image for random VL dataset')
    parser.add_argument('--image-format', type=str, default='RGB', help='Image format for random VL dataset')
    parser.add_argument('--image-num', type=int, default=1, help='Number of images for random VL dataset')
    parser.add_argument('--image-patch-size', type=int, default=28, help='Patch size for image tokenizer, only for local image token calculation')  # noqa: E501

    # Output settings
    parser.add_argument('--outputs-dir', help='Outputs dir.', default='outputs')
    parser.add_argument('--no-timestamp', action='store_true', default=False, help='Do not add timestamp to output directory')  # noqa: E501

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='openqa', help='Specify the dataset')
    parser.add_argument('--dataset-path', type=str, required=False, help='Path to the dataset file')
    parser.add_argument(
        '--dataset-offset',
        type=int,
        default=0,
        help=(
            'Global token-sequence offset for random datasets. '
            'Automatically incremented between runs to avoid KV-cache hits. '
            'Default 0.'
        ),
    )

    # Response settings
    parser.add_argument('--frequency-penalty', type=float, help='The frequency_penalty value', default=None)
    parser.add_argument('--repetition-penalty', type=float, help='The repetition_penalty value', default=None)
    parser.add_argument('--logprobs', action='store_true', help='The logprobs', default=None)
    parser.add_argument(
        '--max-tokens', type=int, nargs='+', help='The maximum number of tokens that can be generated. '
        'Accepts 1 value (fixed) or 2 values min max for uniform sampling per request.', default=2048)
    parser.add_argument(
        '--min-tokens', type=int, help='The minimum number of tokens that can be generated', default=None)
    parser.add_argument('--n-choices', type=int, help='How many completion choices to generate', default=None)
    parser.add_argument('--seed', type=int, help='The random seed', default=None)
    parser.add_argument('--stop', nargs='*', help='The stop tokens', default=None)
    parser.add_argument('--stop-token-ids', nargs='*', help='Set the stop token IDs', default=None)
    parser.add_argument('--stream', action=argparse.BooleanOptionalAction, help='Stream output with SSE', default=True)
    parser.add_argument('--temperature', type=float, help='The sample temperature', default=0.0)
    parser.add_argument('--top-p', type=float, help='Sampling top p', default=None)
    parser.add_argument('--top-k', type=int, help='Sampling top k', default=None)
    parser.add_argument('--extra-args', type=json.loads, default='{}', help='Extra arguments, should in JSON format',)
    parser.add_argument(
        '--tokenize-prompt',
        action='store_true',
        default=False,
        help=(
            'Tokenize prompt client-side and send token IDs directly via /v1/completions, '
            'avoiding the token ID \u2192 text \u2192 token ID re-tokenization inflation. '
            'Requires --tokenizer-path to be set.'
        ),
    )
    # Multi-turn settings
    parser.add_argument(
        '--multi-turn',
        action='store_true',
        default=False,
        help=(
            'Enable multi-turn conversation benchmark mode. '
            'In this mode --number is the total number of conversations to run and '
            '--parallel is the number of conversations executed concurrently. '
            'Use a multi-turn compatible dataset: random_multi_turn, '
            'share_gpt_zh_multi_turn, or share_gpt_en_multi_turn.'
        ),
    )
    parser.add_argument(
        '--min-turns',
        type=int,
        default=1,
        help='Minimum number of user turns per conversation (random_multi_turn only). Default: 1.',
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        default=None,
        help=(
            'Maximum number of user turns per conversation. '
            'For random_multi_turn: required, caps the sampled turn count. '
            'For ShareGPT / custom multi-turn datasets: optional, truncates long conversations. '
            'For swe_smith: upper bound for per-conversation turn count sampling; defaults to --min-turns when unset.'
        ),
    )
    parser.add_argument(
        '--multi-turn-args',
        type=str,
        default=None,
        dest='multi_turn_args',
        help=(
            'Advanced multi-turn conversation parameters for swe_smith (live construction) '
            'as a JSON string. '
            'Example: \'{"first_turn_length": 65000, "subsequent_turn_length": 500, '
            '"chars_per_token": 3.0}\'. '
            'Note: min_turns and max_turns are top-level --min-turns / --max-turns arguments '
            '(per-conversation turn count is sampled from [min_turns, max_turns]).'
        ),
    )
    # yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark LLM service performance.')
    add_argument(parser)
    return parser.parse_args()
