import json

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.benchmarks.live_code_bench.prompts import CodeGenerationPromptConstants


def _make_record() -> dict:
    return {
        'question_content': 'Print the sum of two integers read from stdin.',
        'starter_code': '',
        'public_test_cases': json.dumps([{
            'input': '1 2\n',
            'output': '3\n',
            'testtype': 'stdin'
        }]),
        'private_test_cases': json.dumps([{
            'input': '2 3\n',
            'output': '5\n',
            'testtype': 'stdin'
        }]),
        'metadata': json.dumps({}),
        'contest_date': '2024-01-01T00:00:00',
    }


def _make_adapter() -> LiveCodeBenchAdapter:
    meta: BenchmarkMeta = BENCHMARK_REGISTRY['live_code_bench']
    return LiveCodeBenchAdapter(benchmark_meta=meta)


def test_registered_system_prompt_matches_official_runner() -> None:
    """LiveCodeBench must send the official generic system message (see issue #1541)."""
    assert BENCHMARK_REGISTRY['live_code_bench'].system_prompt == (
        CodeGenerationPromptConstants.SYSTEM_MESSAGE_GENERIC
    )


def test_sample_input_is_system_plus_user_message() -> None:
    adapter = _make_adapter()
    sample = adapter.record_to_sample(_make_record())
    messages = adapter.process_sample_messages_input(sample, subset='release_latest')

    assert len(messages) == 2
    assert isinstance(messages[0], ChatMessageSystem)
    assert messages[0].content == CodeGenerationPromptConstants.SYSTEM_MESSAGE_GENERIC
    assert isinstance(messages[1], ChatMessageUser)
    assert '### Question:' in messages[1].content
    assert '### Format:' in messages[1].content
