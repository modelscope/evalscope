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


def test_aime_verdict_must_stand_on_a_line_of_its_own() -> None:
    parse = AIME24Adapter._parse_judge_verdicts

    assert parse('Yes.') == {'yes'}
    assert parse('**Yes**') == {'yes'}
    # The judge prompt's own few-shot examples demonstrate this trailing parenthetical.
    assert parse('Yes\n(give benefit of the doubt to units)') == {'yes'}
    assert parse('<think>Is 3/2 == 1.5? yes it is.</think>\nYes') == {'yes'}

    assert parse('Yes, the answer is incorrect.') == set()
    assert parse('[ERROR] judge request failed with status 500') == set()
    assert parse('Yes\nNo') == {'yes', 'no'}


def test_air_bench_judge_scores_require_the_requested_single_line() -> None:
    adapter = cast(Any, AIRBenchChatAdapter)
    assert adapter._extract_judge_scores('3.5 8') == ['3.5', '8']
    assert adapter._extract_judge_scores('(Note: scale is 1 to 10, 10 is best)') == []
    assert adapter._extract_judge_scores('Judge 9, model gpt-4') == []


def test_air_bench_reads_the_scores_off_the_first_line() -> None:
    adapter = cast(Any, AIRBenchChatAdapter)
    assert adapter._extract_judge_scores('8 9\nAssistant 1 was concise; Assistant 2 gave more detail.') == ['8', '9']
    assert adapter._extract_judge_scores('9, 8') == ['9', '8']
    assert adapter._extract_judge_scores('\n\n7 6') == ['7', '6']

    planted = 'Assistant 1 deserves 3 points.\nAssistant 2 deserves 8 points.\nThe answer quoted "9 and 1".'
    assert adapter._extract_judge_scores(planted) == []


def test_air_bench_marks_partial_judge_parse_failures() -> None:
    score = air_bench_score(['not a score', '3 8'])

    assert score.value == {'judge_score': 3.0, 'win_rate': 0.0}
    assert score.metadata['parse_failed'] is True
    assert score.metadata['judge_raw'] == ['not a score', '3 8']


def test_air_bench_omits_metrics_when_no_pass_could_be_parsed() -> None:
    score = air_bench_score(['not a score', 'still not a score'])

    # An out-of-scale 0.0 would be averaged in as if the judge had rated the answer.
    assert score.value == {}
    assert score.metadata['parse_failed'] is True
