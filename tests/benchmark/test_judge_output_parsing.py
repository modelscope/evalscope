from types import SimpleNamespace
from typing import Any, cast

from evalscope.benchmarks.aime.aime_adapter import AIME24Adapter
from evalscope.benchmarks.air_bench.air_bench_chat_adapter import AIRBenchChatAdapter


class StubJudge:

    model_id = 'stub-judge'

    def __init__(self, response: str) -> None:
        self.response = response

    def judge(self, prompt: str) -> str:
        return self.response


class SequenceJudge:

    model_id = 'stub-judge'

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def judge(self, prompt: str, system_prompt: str) -> str:
        return next(self.responses)


def aime_score(judge_response: str) -> Any:
    adapter = cast(Any, object.__new__(AIME24Adapter))
    adapter._llm_judge = StubJudge(judge_response)
    adapter._task_config = SimpleNamespace(judge_strategy='llm')
    return adapter.llm_match_score('prediction', 'prediction', 'reference', None)


def air_bench_score(responses: list[str]) -> Any:
    adapter = cast(Any, object.__new__(AIRBenchChatAdapter))
    adapter._llm_judge = SequenceJudge(responses)
    adapter._task_config = SimpleNamespace(judge_strategy='llm')
    adapter._benchmark_meta = SimpleNamespace(get_extra_params=lambda: {'do_swap': True})
    task_state = SimpleNamespace(metadata={'meta_info': ''}, input_text='question')
    return adapter.llm_match_score('prediction', 'prediction', 'reference', task_state)


def test_aime_llm_judge_accepts_only_a_yes_verdict() -> None:
    assert aime_score(' Yes\n').value['acc'] == 1.0

    malformed = aime_score('No, they are not equivalent. Yes would be wrong.')
    assert malformed.value['acc'] == 0.0
    assert malformed.metadata['parse_failed'] is True


def test_air_bench_judge_scores_require_the_requested_single_line() -> None:
    adapter = cast(Any, AIRBenchChatAdapter)
    assert adapter._extract_judge_scores('3.5 8') == ['3.5', '8']
    assert adapter._extract_judge_scores('(Note: scale is 1 to 10, 10 is best)') == []
    assert adapter._extract_judge_scores('Judge 9, model gpt-4') == []


def test_air_bench_marks_partial_judge_parse_failures() -> None:
    score = air_bench_score(['not a score', '3 8'])

    assert score.value == {'judge_score': 3.0, 'win_rate': 0.0}
    assert score.metadata['parse_failed'] is True
    assert score.metadata['judge_raw'] == ['not a score', '3 8']
