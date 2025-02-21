import time
import torch
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass
class BenchmarkData:
    request: Any = None
    start_time: float = field(default_factory=time.perf_counter)
    completed_time: float = 0.0
    chunk_times: List[float] = field(default_factory=list)
    success: bool = False
    response_messages: List[Any] = field(default_factory=list)

    # late init
    query_latency: float = 0.0
    first_chunk_latency: float = 0.0
    n_chunks: int = 0
    n_chunks_time: float = 0.0
    max_gpu_memory_cost = 0
    time_per_output_token: float = 0.0

    prompt_tokens = None
    completion_tokens = None

    def _calculate_query_stream_metric(self) -> Tuple[float, int, float]:
        self.query_latency = self.completed_time - self.start_time
        if len(self.chunk_times) > 1:
            self.first_chunk_latency = self.chunk_times[0] - self.start_time
            self.n_chunks = len(self.chunk_times) - 2
            self.n_chunks_time = self.chunk_times[-2] - self.chunk_times[0]
        else:
            self.first_chunk_latency = self.query_latency
            self.n_chunks = 1
            self.n_chunks_time = self.query_latency
        self.time_per_output_token = self.query_latency / self.completion_tokens

    def _calculate_tokens(self, api_plugin):
        self.prompt_tokens, self.completion_tokens = \
            api_plugin.parse_responses(self.response_messages, request=self.request)

    def update_gpu_usage(self):
        total_memory = 0
        for i in range(torch.cuda.device_count()):
            total_memory += (torch.cuda.max_memory_allocated(i) / 2**30)  # GB
        self.max_gpu_memory_cost = max(self.max_gpu_memory_cost, total_memory)


@dataclass
class BenchmarkMetrics:
    concurrency: int = 0
    n_succeed_queries: int = 0
    n_failed_queries: int = 0
    total_first_chunk_latency: float = 0.0
    total_latency: float = 0.0
    n_total_chunks: int = 0
    n_total_prompt_tokens: int = 0
    n_total_completion_tokens: int = 0
    total_chunks_time: float = 0.0
    start_time: Optional[float] = None
    total_time: float = 1.0
    n_total_queries: int = 0
    n_time_per_output_token: float = 0.0

    avg_first_chunk_latency: float = -1
    avg_latency: float = -1
    n_avg_chunks: float = -1
    avg_chunk_time: float = -1
    avg_prompt_tokens: float = -1
    avg_completion_tokens: float = -1
    avg_token_per_seconds: float = -1
    avg_time_per_token: float = -1
    qps: float = -1

    def update_metrics(self, benchmark_data: BenchmarkData, api_plugin):
        self.n_total_queries += 1
        if self.start_time is None:
            self.start_time = benchmark_data.start_time
        self.total_time = time.perf_counter() - self.start_time

        if benchmark_data.success:
            self.n_succeed_queries += 1

            benchmark_data._calculate_tokens(api_plugin)
            self.n_total_prompt_tokens += benchmark_data.prompt_tokens
            self.n_total_completion_tokens += benchmark_data.completion_tokens

            benchmark_data._calculate_query_stream_metric()
            self.total_latency += benchmark_data.query_latency
            self.total_first_chunk_latency += benchmark_data.first_chunk_latency
            self.n_total_chunks += benchmark_data.n_chunks
            self.total_chunks_time += benchmark_data.n_chunks_time
            self.n_time_per_output_token += benchmark_data.time_per_output_token
        else:
            self.n_failed_queries += 1

        self.calculate_averages()

    def calculate_averages(self):
        if self.n_succeed_queries == 0:
            return
        try:
            self.avg_first_chunk_latency = self.total_first_chunk_latency / self.n_succeed_queries
            self.avg_latency = self.total_latency / self.n_succeed_queries
            self.n_avg_chunks = self.n_total_chunks / self.n_succeed_queries
            self.avg_chunk_time = self.total_chunks_time / self.n_total_chunks
            self.avg_prompt_tokens = self.n_total_prompt_tokens / self.n_succeed_queries
            self.avg_completion_tokens = self.n_total_completion_tokens / self.n_succeed_queries
            self.avg_token_per_seconds = self.n_total_completion_tokens / self.total_time
            self.avg_time_per_token = self.n_time_per_output_token / self.n_succeed_queries
            self.qps = self.n_succeed_queries / self.total_time
        except ZeroDivisionError as e:
            logger.exception(e)
            return

    def create_message(self, default_ndigits=3):
        message = {
            'Time taken for tests (s)': round(self.total_time, default_ndigits),
            'Number of concurrency': self.concurrency,
            'Total requests': int(self.n_total_queries),
            'Succeed requests': self.n_succeed_queries,
            'Failed requests': self.n_failed_queries,
            'Throughput(average tokens/s)': round(self.avg_token_per_seconds, default_ndigits),
            'Average QPS': round(self.qps, default_ndigits),
            'Average latency (s)': round(self.avg_latency, default_ndigits),
            'Average time to first token (s)': round(self.avg_first_chunk_latency, default_ndigits),
            'Average time per output token (s)': round(self.avg_time_per_token, default_ndigits),
            'Average input tokens per request': round(self.avg_prompt_tokens, default_ndigits),
            'Average output tokens per request': round(self.avg_completion_tokens, default_ndigits),
            'Average package latency (s)': round(self.avg_chunk_time, default_ndigits),
            'Average package per request': round(self.n_avg_chunks, default_ndigits),
        }
        return message
