from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass
class BenchmarkData:
    request: str = None  # json serialized request body
    start_time: float = 0.0
    completed_time: float = 0.0
    chunk_times: List[float] = field(default_factory=list)
    success: bool = False
    response_messages: List[Any] = field(default_factory=list)

    # late init
    query_latency: float = 0.0
    first_chunk_latency: float = 0.0
    max_gpu_memory_cost = 0
    time_per_output_token: float = 0.0
    inter_chunk_latency: List[float] = field(default_factory=list)

    # response content
    generated_text: str = ''
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    def _calculate_tokens(self, api_plugin):
        if self.prompt_tokens is None or self.completion_tokens is None:
            self.prompt_tokens, self.completion_tokens = api_plugin.parse_responses(
                self.response_messages, request=self.request
            )

        # Calculate time per output token
        if self.completion_tokens and self.completion_tokens > 1:
            # tpot = (latency - ttft) / (output_len - 1)
            self.time_per_output_token = (self.query_latency - self.first_chunk_latency) / (self.completion_tokens - 1)

        # Ensure inter-chunk latency is available (compute from chunk_times if needed)
        if not self.inter_chunk_latency and self.chunk_times:
            self.inter_chunk_latency = [t2 - t1 for t1, t2 in zip(self.chunk_times[:-1], self.chunk_times[1:])]

    def update_gpu_usage(self):
        if check_import('torch', raise_warning=False):
            import torch
            total_memory = 0
            for i in range(torch.cuda.device_count()):
                total_memory += (torch.cuda.max_memory_allocated(i) / 2**30)  # GB
            self.max_gpu_memory_cost = max(self.max_gpu_memory_cost, total_memory)


class Metrics:
    TIME_TAKEN_FOR_TESTS = 'Time taken for tests (s)'
    NUMBER_OF_CONCURRENCY = 'Number of concurrency'
    TOTAL_REQUESTS = 'Total requests'
    SUCCEED_REQUESTS = 'Succeed requests'
    FAILED_REQUESTS = 'Failed requests'
    OUTPUT_TOKEN_THROUGHPUT = 'Output token throughput (tok/s)'
    TOTAL_TOKEN_THROUGHPUT = 'Total token throughput (tok/s)'
    REQUEST_THROUGHPUT = 'Request throughput (req/s)'
    AVERAGE_LATENCY = 'Average latency (s)'
    AVERAGE_TIME_TO_FIRST_TOKEN = 'Average time to first token (s)'
    AVERAGE_TIME_PER_OUTPUT_TOKEN = 'Average time per output token (s)'
    AVERAGE_INTER_TOKEN_LATENCY = 'Average inter-token latency (s)'
    AVERAGE_INPUT_TOKENS_PER_REQUEST = 'Average input tokens per request'
    AVERAGE_OUTPUT_TOKENS_PER_REQUEST = 'Average output tokens per request'


@dataclass
class BenchmarkMetrics:
    concurrency: int = 0
    n_succeed_queries: int = 0
    n_failed_queries: int = 0
    total_first_chunk_latency: float = 0.0
    total_latency: float = 0.0
    n_total_prompt_tokens: int = 0
    n_total_completion_tokens: int = 0
    start_time: Optional[float] = None
    last_completed_time: Optional[float] = None
    total_time: float = 1.0
    n_total_queries: int = 0
    n_time_per_output_token: float = 0.0
    n_total_inter_token_latency: List[float] = field(default_factory=list)

    avg_first_chunk_latency: float = -1
    avg_latency: float = -1
    avg_prompt_tokens: float = -1
    avg_completion_tokens: float = -1
    avg_input_token_per_seconds: float = -1
    avg_output_token_per_seconds: float = -1
    avg_total_token_per_seconds: float = -1
    avg_time_per_token: float = -1
    avg_inter_token_latency: float = -1
    qps: float = -1

    def update_metrics(self, benchmark_data: BenchmarkData, api_plugin):
        self.n_total_queries += 1

        if benchmark_data.success:
            self.n_succeed_queries += 1

            benchmark_data._calculate_tokens(api_plugin)
            self.n_total_prompt_tokens += benchmark_data.prompt_tokens
            self.n_total_completion_tokens += benchmark_data.completion_tokens

            self.total_latency += benchmark_data.query_latency
            self.total_first_chunk_latency += benchmark_data.first_chunk_latency
            self.n_time_per_output_token += benchmark_data.time_per_output_token
            self.n_total_inter_token_latency += benchmark_data.inter_chunk_latency
        else:
            self.n_failed_queries += 1

        self.update_total_time(benchmark_data)
        self.calculate_averages()

    def update_total_time(self, benchmark_data: BenchmarkData):
        # Use the earliest start_time seen so far
        if self.start_time is None:
            self.start_time = benchmark_data.start_time
        else:
            self.start_time = min(self.start_time, benchmark_data.start_time)
        # Track the latest completion time
        if self.last_completed_time is None:
            self.last_completed_time = benchmark_data.completed_time
        else:
            self.last_completed_time = max(self.last_completed_time, benchmark_data.completed_time)
        # Compute total_time from request lifecycle timestamps to avoid consumer overhead
        if self.start_time is not None and self.last_completed_time is not None:
            self.total_time = max(self.last_completed_time - self.start_time, 0.0)

    def calculate_averages(self):
        if self.n_succeed_queries == 0:
            return
        try:
            self.avg_first_chunk_latency = self.total_first_chunk_latency / self.n_succeed_queries
            self.avg_latency = self.total_latency / self.n_succeed_queries
            self.avg_prompt_tokens = self.n_total_prompt_tokens / self.n_succeed_queries
            self.avg_completion_tokens = self.n_total_completion_tokens / self.n_succeed_queries
            self.avg_input_token_per_seconds = self.n_total_prompt_tokens / self.total_first_chunk_latency
            self.avg_output_token_per_seconds = self.n_total_completion_tokens / self.total_time
            self.avg_total_token_per_seconds = (
                self.n_total_prompt_tokens + self.n_total_completion_tokens
            ) / self.total_time
            self.avg_time_per_token = self.n_time_per_output_token / self.n_succeed_queries
            self.avg_inter_token_latency = sum(self.n_total_inter_token_latency) / len(
                self.n_total_inter_token_latency
            ) if self.n_total_inter_token_latency else 0.0
            self.qps = self.n_succeed_queries / self.total_time
        except ZeroDivisionError as e:
            logger.exception(e)
            return

    def create_message(self, default_ndigits=4):
        message = {
            Metrics.TIME_TAKEN_FOR_TESTS: round(self.total_time, default_ndigits),
            Metrics.NUMBER_OF_CONCURRENCY: self.concurrency,
            Metrics.TOTAL_REQUESTS: int(self.n_total_queries),
            Metrics.SUCCEED_REQUESTS: self.n_succeed_queries,
            Metrics.FAILED_REQUESTS: self.n_failed_queries,
            Metrics.OUTPUT_TOKEN_THROUGHPUT: round(self.avg_output_token_per_seconds, default_ndigits),
            Metrics.TOTAL_TOKEN_THROUGHPUT: round(self.avg_total_token_per_seconds, default_ndigits),
            Metrics.REQUEST_THROUGHPUT: round(self.qps, default_ndigits),
            Metrics.AVERAGE_LATENCY: round(self.avg_latency, default_ndigits),
            Metrics.AVERAGE_TIME_TO_FIRST_TOKEN: round(self.avg_first_chunk_latency, default_ndigits),
            Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN: round(self.avg_time_per_token, default_ndigits),
            Metrics.AVERAGE_INTER_TOKEN_LATENCY: round(self.avg_inter_token_latency, default_ndigits),
            Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST: round(self.avg_prompt_tokens, default_ndigits),
            Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST: round(self.avg_completion_tokens, default_ndigits),
        }
        return message
