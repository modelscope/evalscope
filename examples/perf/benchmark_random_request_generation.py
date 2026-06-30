"""Benchmark random dataset request generation without sending HTTP requests.

This script exercises the same ``get_requests()`` path used by ``evalscope perf``
and measures only local request construction.  It is intended for comparing
serial generation (``--num-workers 1``) with multiprocessing generation for the
``random`` dataset.

Example:
    python examples/perf/benchmark_random_request_generation.py \
        --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
        --numbers 128 512 \
        --prompt-lengths 2048 8192 \
        --workers 1 2 4 8 0
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.benchmark import get_requests
from evalscope.perf.plugin.datasets.random_dataset import RandomDatasetPlugin
from evalscope.perf.utils.worker_util import resolve_dataset_generation_workers

_BENCHMARK_URL = 'http://127.0.0.1:8000/v1/chat/completions'


class _NoopApiPlugin:
    """Build request dictionaries without touching the network."""

    def build_request(self, messages: Any) -> Dict[str, Any]:
        return {'messages': messages}


@dataclass
class _BenchmarkResult:
    number: int
    prompt_length: int
    configured_workers: int
    resolved_workers: int
    repeat: int
    seconds: float
    requests_per_second: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark random request generation throughput.')
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B-Instruct',
        help='Tokenizer path or model id used by the random dataset.',
    )
    parser.add_argument(
        '--numbers',
        type=int,
        nargs='+',
        default=[128, 512],
        help='Request counts to benchmark.',
    )
    parser.add_argument(
        '--prompt-lengths',
        type=int,
        nargs='+',
        default=[2048, 8192],
        help='Fixed prompt lengths to benchmark.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 0],
        help='Configured --num-workers values. Use 0 for auto.',
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='Number of repeats per case.',
    )
    parser.add_argument(
        '--apply-chat-template',
        action='store_true',
        default=False,
        help='Generate chat-template-shaped messages instead of raw text prompts.',
    )
    parser.add_argument(
        '--tokenize-prompt',
        action='store_true',
        default=False,
        help='Benchmark token-id prompt generation for the completions endpoint.',
    )
    return parser.parse_args()


def _make_arguments(
    tokenizer_path: str,
    number: int,
    prompt_length: int,
    workers: int,
    apply_chat_template: bool,
    tokenize_prompt: bool,
) -> Arguments:
    return Arguments(
        model='request-generation-benchmark',
        url=_BENCHMARK_URL,
        dataset='random',
        tokenizer_path=tokenizer_path,
        number=number,
        parallel=1,
        num_workers=workers,
        min_prompt_length=prompt_length,
        max_prompt_length=prompt_length,
        max_tokens=1,
        apply_chat_template=apply_chat_template,
        tokenize_prompt=tokenize_prompt,
    )


def _random_dataset_supports_parallel(args: Arguments) -> bool:
    plugin = object.__new__(RandomDatasetPlugin)
    plugin.query_parameters = args
    plugin.number = args.total_count
    return plugin.supports_parallel_message_generation(args.total_count)


async def _consume_requests(args: Arguments) -> int:
    count = 0
    async for _request, _is_warmup in get_requests(args, _NoopApiPlugin()):
        count += 1
    return count


def _run_case(
    tokenizer_path: str,
    number: int,
    prompt_length: int,
    workers: int,
    repeat: int,
    apply_chat_template: bool,
    tokenize_prompt: bool,
) -> _BenchmarkResult:
    args = _make_arguments(
        tokenizer_path=tokenizer_path,
        number=number,
        prompt_length=prompt_length,
        workers=workers,
        apply_chat_template=apply_chat_template,
        tokenize_prompt=tokenize_prompt,
    )
    resolved_workers = resolve_dataset_generation_workers(
        args=args,
        total_count=args.total_count,
        supports_parallel_generation=_random_dataset_supports_parallel(args),
    )

    start = time.perf_counter()
    count = asyncio.run(_consume_requests(args))
    seconds = time.perf_counter() - start
    if count != args.total_count:
        raise RuntimeError(f'Expected {args.total_count} requests, got {count}.')

    return _BenchmarkResult(
        number=number,
        prompt_length=prompt_length,
        configured_workers=workers,
        resolved_workers=resolved_workers,
        repeat=repeat,
        seconds=seconds,
        requests_per_second=count / seconds,
    )


def _run_matrix(args: argparse.Namespace) -> List[_BenchmarkResult]:
    results = []
    for number in args.numbers:
        for prompt_length in args.prompt_lengths:
            for workers in args.workers:
                for repeat_index in range(args.repeat):
                    results.append(
                        _run_case(
                            tokenizer_path=args.tokenizer_path,
                            number=number,
                            prompt_length=prompt_length,
                            workers=workers,
                            repeat=repeat_index + 1,
                            apply_chat_template=args.apply_chat_template,
                            tokenize_prompt=args.tokenize_prompt,
                        )
                    )
    return results


def _print_results(results: List[_BenchmarkResult]) -> None:
    print('number,prompt_length,configured_workers,resolved_workers,repeat,seconds,requests_per_second')
    for result in results:
        print(
            f'{result.number},{result.prompt_length},{result.configured_workers},'
            f'{result.resolved_workers},{result.repeat},{result.seconds:.3f},'
            f'{result.requests_per_second:.2f}'
        )


def main() -> None:
    args = _parse_args()
    results = _run_matrix(args)
    _print_results(results)


if __name__ == '__main__':
    main()
