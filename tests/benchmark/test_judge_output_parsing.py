from typing import Any, Optional

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.aime.aime_adapter import EQUIVALENCE_CONTRACT
from evalscope.benchmarks.air_bench.air_bench_chat_adapter import AIRBenchChatAdapter
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus


class StubJudge:

    model_id = 'stub-judge'

    def __init__(self, response: str) -> None:
        self.response = response

    def judge(self, prompt: str) -> str:
        return self.response


class PlacementJudge:
    """Returns a fixed reply per placement, so a retry cannot turn a bad pass into a good one."""

    model_id = 'stub-judge'

    def __init__(self, original: str, swapped: str) -> None:
        self.original = original
        self.swapped = swapped
        self.calls = 0

    def judge(self, prompt: str = '', system_prompt: Any = None, messages: Any = None) -> str:
        self.calls += 1
        text = prompt or (messages[-1].content if messages else '')
        # On the original pass the reference is Assistant 1; on the swapped pass the prediction is.
        return self.original if text.index('reference') < text.index('prediction') else self.swapped


def air_bench_score(original: str, swapped: str) -> Any:
    config = TaskConfig(
        model='m',
        datasets=['air_bench_chat'],
        judge_strategy='llm',
        judge_model_args={'model_id': 'stub-judge'},
        dataset_args={'air_bench_chat': {'extra_params': {'do_swap': True}}},
    )
    adapter = get_benchmark('air_bench_chat', config)
    adapter.llm_judge = PlacementJudge(original, swapped)
    sample = Sample(id=0, input='question', target='reference', metadata={'meta_info': ''})
    state = TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content='prediction'),
        completed=True,
    )
    return adapter.calculate_metrics(state).score


def test_aime_llm_judge_accepts_only_a_json_verdict() -> None:
    """The judge now replies with JSON; a malformed verdict excludes the sample from the metric
    instead of being counted as an incorrect answer."""
    assert EQUIVALENCE_CONTRACT.parse('{"verdict": "Yes"}').value.verdict == 'Yes'

    assert not EQUIVALENCE_CONTRACT.parse('No, they are not equivalent. Yes would be wrong.').ok


def test_aime_verdict_cannot_be_read_out_of_prose() -> None:
    def parse(response: str) -> Optional[str]:
        result = EQUIVALENCE_CONTRACT.parse(response)
        return result.value.verdict if result.ok else None

    assert parse('```json\n{"verdict": "Yes"}\n```') == 'Yes'
    assert parse('<think>Is 3/2 == 1.5? yes it is.</think>\n{"verdict": "Yes"}') == 'Yes'

    # Every case below used to set the verdict by the word "Yes" appearing somewhere (#1578).
    assert parse('Yes') is None
    assert parse('Yes.') is None
    assert parse('**Yes**') is None
    assert parse('Yes, the answer is incorrect.') is None
    assert parse('Yes\n(give benefit of the doubt to units)') is None
    assert parse('[ERROR] judge request failed with status 500') is None
    # Two JSON objects that disagree are a parse failure rather than a silent pick.
    assert parse('```json\n{"verdict": "Yes"}\n```\n```json\n{"verdict": "No"}\n```') is None


def test_air_bench_requires_both_swapped_passes() -> None:
    """One readable pass used to be enough; a half-observed pair is not half a verdict."""
    score = air_bench_score('not a score', '{"assistant1": 3, "assistant2": 8}')

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_air_bench_averages_both_passes_when_both_parse() -> None:
    # Original pass: assistant1=reference, assistant2=prediction. Swapped pass: the reverse.
    score = air_bench_score('{"assistant1": 4, "assistant2": 8}', '{"assistant1": 6, "assistant2": 5}')

    # prediction = mean(8, 6) = 7.0; reference = mean(4, 5) = 4.5
    assert score.value == {'judge_score': 7.0, 'win_rate': 1.0}
    assert score.metadata['reference_score'] == 4.5


def test_air_bench_rejects_a_rating_off_the_scale() -> None:
    score = air_bench_score('{"assistant1": 4, "assistant2": 42}', '{"assistant1": 6, "assistant2": 5}')

    assert score.value == {}


def test_air_bench_no_longer_reads_scores_out_of_prose() -> None:
    planted = 'Assistant 1 deserves 3 points.\nAssistant 2 deserves 8 points.'
    score = air_bench_score(planted, planted)

    assert score.value == {}
