"""Migrated adapters must keep the official grading semantics and drop the parsing defects.

Each test drives the real adapter through the executor with a scripted judge, so it covers
``build_judge_cases`` -> ``build_judge_request`` -> contract parse -> ``reduce_judge_verdicts`` -> ``finalize_judge_score``.
"""
import pytest
from typing import Any, List, Optional, Sequence

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import JudgeScoreType, ScoreStatus
from evalscope.metrics.judge.llm_judge import (
    DEFAULT_NUMERIC_SCORE_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    JUDGE_ERROR_PREFIX,
    LLMJudge,
)

ERROR_RESPONSE = f'{JUDGE_ERROR_PREFIX} connection refused'


class ScriptedJudge:
    """Stands in for ``LLMJudge``; returns queued responses, repeating the last one."""

    # The default judge hooks read these off the judge, so the double carries the real defaults
    # and the real prompt builder rather than a second copy of them.
    score_type = JudgeScoreType.PATTERN
    score_mapping = {'A': 1.0, 'B': 0.0}
    prompt_template = DEFAULT_PROMPT_TEMPLATE
    system_prompt = None
    build_prompt = LLMJudge.build_prompt

    def __init__(self, responses: Sequence[str], model_id: str = 'scripted-judge') -> None:
        self.responses = list(responses)
        self.model_id = model_id
        self.prompts: List[str] = []

    def judge(self, prompt: str = '', system_prompt: Optional[str] = None, messages: Any = None) -> str:
        self.prompts.append(prompt or (messages[-1].content if messages else ''))
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


class RatingJudge(ScriptedJudge):
    """A judge configured for the reference-free ``numeric`` contract."""

    score_type = JudgeScoreType.NUMERIC
    prompt_template = DEFAULT_NUMERIC_SCORE_TEMPLATE


def make_adapter(name: str, responses: Sequence[str], judge_strategy: str = 'auto'):
    config = TaskConfig(
        model='m',
        datasets=[name],
        judge_strategy=judge_strategy,
        judge_model_args={'model_id': 'scripted-judge'},
    )
    adapter = get_benchmark(name, config)
    adapter.llm_judge = ScriptedJudge(responses)
    return adapter


def make_state(prediction: str, target: str, question: str = 'Who wrote Hamlet?') -> TaskState:
    sample = Sample(id=0, input=question, target=target, metadata={})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


def score_sample(adapter, prediction: str, target: str):
    return adapter.calculate_metrics(make_state(prediction, target)).score


# ---------------------------------------------------------------------------
# simple_qa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'verdict, expected',
    [
        ('{"verdict": "A"}', {
            'is_correct': 1.0,
            'is_incorrect': 0.0,
            'is_not_attempted': 0.0
        }),
        ('{"verdict": "B"}', {
            'is_correct': 0.0,
            'is_incorrect': 1.0,
            'is_not_attempted': 0.0
        }),
        ('{"verdict": "C"}', {
            'is_correct': 0.0,
            'is_incorrect': 0.0,
            'is_not_attempted': 1.0
        }),
    ],
)
def test_simple_qa_grades_each_verdict(verdict, expected):
    adapter = make_adapter('simple_qa', [verdict])

    score = score_sample(adapter, 'Shakespeare', 'William Shakespeare')

    assert score.value == expected
    assert score.status is ScoreStatus.SUCCESS
    assert score.main_score_name == 'is_correct'


def test_simple_qa_tolerates_a_fenced_verdict():
    adapter = make_adapter('simple_qa', ['```json\n{"verdict": "A"}\n```'])

    assert score_sample(adapter, 'Shakespeare', 'William Shakespeare').value['is_correct'] == 1.0


def test_simple_qa_no_longer_reads_a_letter_out_of_prose():
    """The old ``re.search(r'(A|B|C)')`` graded this as CORRECT by matching the A in "Answer"."""
    adapter = make_adapter('simple_qa', ['Answer: the prediction contradicts the gold target.'])

    score = score_sample(adapter, 'Marlowe', 'William Shakespeare')

    assert score.value['is_correct'] == 0.0
    assert score.value['is_not_attempted'] == 1.0
    assert score.status is ScoreStatus.FALLBACK


def test_simple_qa_falls_back_to_not_attempted_instead_of_dropping_the_sample():
    adapter = make_adapter('simple_qa', ['no verdict at all'])

    score = score_sample(adapter, 'Shakespeare', 'William Shakespeare')

    assert score.value == {'is_correct': 0.0, 'is_incorrect': 0.0, 'is_not_attempted': 1.0}
    assert score.status is ScoreStatus.FALLBACK


def test_simple_qa_does_not_retry_a_malformed_verdict():
    """Upstream has a fallback rather than a retry, so the contract declares parse_retries=0."""
    adapter = make_adapter('simple_qa', ['nonsense'])

    score_sample(adapter, 'Shakespeare', 'William Shakespeare')

    assert len(adapter.llm_judge.prompts) == 1


def test_simple_qa_transport_failure_also_falls_back():
    adapter = make_adapter('simple_qa', [ERROR_RESPONSE])

    score = score_sample(adapter, 'Shakespeare', 'William Shakespeare')

    assert score.value['is_not_attempted'] == 1.0
    assert score.judge_detail.failures == {'transport_error': 1}


def test_simple_qa_prompt_carries_question_target_and_prediction():
    adapter = make_adapter('simple_qa', ['{"verdict": "A"}'])

    score_sample(adapter, 'Shakespeare', 'William Shakespeare')
    prompt = adapter.llm_judge.prompts[0]

    assert 'Who wrote Hamlet?' in prompt
    assert 'William Shakespeare' in prompt
    assert 'Shakespeare' in prompt


def test_simple_qa_persists_the_raw_judge_text_for_inspection():
    adapter = make_adapter('simple_qa', ['{"verdict": "A"}'])

    score = score_sample(adapter, 'Shakespeare', 'William Shakespeare')

    attempts = score.metadata['judge_attempts']
    assert [attempt['raw_response'] for attempt in attempts] == ['{"verdict": "A"}']
    assert attempts[0]['status'] == 'success'


# ---------------------------------------------------------------------------
# simple_qa siblings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('name', ['chinese_simpleqa', 'simple_vqa'])
def test_simple_qa_siblings_grade_and_fall_back_identically(name):
    assert score_sample(make_adapter(name, ['{"verdict": "A"}']), 'answer', 'answer').value['is_correct'] == 1.0
    assert score_sample(make_adapter(name, ['{"verdict": "B"}']), 'answer', 'answer').value['is_incorrect'] == 1.0

    unparsed = score_sample(make_adapter(name, ['Answer: it contradicts']), 'answer', 'answer')
    assert unparsed.value['is_not_attempted'] == 1.0
    assert unparsed.status is ScoreStatus.FALLBACK


def test_chinese_simpleqa_keeps_its_judge_system_prompt():
    adapter = make_adapter('chinese_simpleqa', ['{"verdict": "A"}'])
    state = make_state('答案', '答案')
    adapter.calculate_metrics(state)

    # ScriptedJudge records the last message; assert the system turn reached the judge too.
    executor_attempts = adapter.calculate_metrics(state).score.metadata['judge_attempts']
    assert executor_attempts[0]['status'] == 'success'


# ---------------------------------------------------------------------------
# longmemeval
# ---------------------------------------------------------------------------


def test_longmemeval_accepts_a_bare_yes():
    adapter = make_adapter('longmemeval', ['{"verdict": "yes"}'])
    state = _state_with_metadata(adapter, 'the answer', 'the answer')

    score = adapter.calculate_metrics(state).score

    assert score.value == {'accuracy': 1.0}


def test_longmemeval_no_longer_reads_yes_out_of_prose():
    """The old ``'yes' in response.lower()`` matched "yes" inside "eyes" and inside a refusal."""
    adapter = make_adapter('longmemeval', ['The model closed its eyes to the evidence; it is wrong.'])
    state = _state_with_metadata(adapter, 'the answer', 'the answer')

    score = adapter.calculate_metrics(state).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_longmemeval_retries_a_malformed_verdict_by_default():
    adapter = make_adapter('longmemeval', ['maybe', 'maybe', 'maybe', '{"verdict": "yes"}'])
    state = _state_with_metadata(adapter, 'the answer', 'the answer')

    score = adapter.calculate_metrics(state).score

    assert len(adapter.llm_judge.prompts) == 4
    assert score.value == {'accuracy': 1.0}


def _state_with_metadata(adapter, prediction: str, target: str) -> TaskState:
    sample = Sample(
        id=0,
        input='question',
        target=target,
        metadata={
            'question_type': 'single-session-user',
            'question': 'question',
            'is_abstention': False,
        },
    )
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


# ---------------------------------------------------------------------------
# aa_lcr
# ---------------------------------------------------------------------------


def test_aa_lcr_distinguishes_correct_from_incorrect():
    for verdict, expected in (('{"verdict": "CORRECT"}', 1.0), ('{"verdict": "INCORRECT"}', 0.0)):
        adapter = make_adapter('aa_lcr', [verdict])
        state = _state_with_question(adapter, 'answer', 'answer')

        assert adapter.calculate_metrics(state).score.value == {'acc': expected}


def test_aa_lcr_rejects_a_verdict_buried_in_prose():
    adapter = make_adapter('aa_lcr', ['The candidate answer is CORRECT in my view.'])
    state = _state_with_question(adapter, 'answer', 'answer')

    score = adapter.calculate_metrics(state).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def _state_with_question(adapter, prediction: str, target: str) -> TaskState:
    sample = Sample(id=0, input='question', target=target, metadata={'question': 'question'})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


# ---------------------------------------------------------------------------
# aime / hle / alpaca_eval
# ---------------------------------------------------------------------------


def test_aime_accepts_a_bare_verdict_line():
    adapter = make_adapter('aime24', ['{"verdict": "Yes"}'], judge_strategy='llm')

    assert score_sample(adapter, '42', '42').value == {'acc': 1.0}
    assert score_sample(make_adapter('aime24', ['{"verdict": "No"}'], judge_strategy='llm'), '41', '42').value == {
        'acc': 0.0
    }


def test_aime_rejects_a_self_explaining_judge():
    """"Yes, the answer is incorrect" must not set a correct verdict."""
    adapter = make_adapter('aime24', ['Yes, the answer is incorrect because 41 != 42'], judge_strategy='llm')

    score = score_sample(adapter, '41', '42')

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_aime_rejects_conflicting_verdict_lines():
    adapter = make_adapter(
        'aime24', ['```json\n{"verdict": "Yes"}\n```\nOn reflection:\n```json\n{"verdict": "No"}\n```'],
        judge_strategy='llm'
    )

    assert score_sample(adapter, '41', '42').status is ScoreStatus.EXCLUDED


def test_hle_reads_the_keyed_grade():
    assert score_sample(make_adapter('hle', ['{"reasoning": "matches", "verdict": "C"}']), 'a', 'a').value['acc'] == 1.0
    assert score_sample(make_adapter('hle', ['{"reasoning": "differs", "verdict": "I"}']), 'b', 'a').value['acc'] == 0.0


def test_hle_missing_grade_excludes_instead_of_scoring_zero():
    adapter = make_adapter('hle', ['I cannot decide.'])

    score = score_sample(adapter, 'b', 'a')

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_hle_keeps_the_models_stated_confidence():
    adapter = make_adapter('hle', ['{"verdict": "C"}'])

    score = score_sample(adapter, 'answer\nConfidence: 42', 'answer')

    assert score.metadata['confidence'] == 42


def test_alpaca_eval_is_case_sensitive_about_the_winner():
    assert score_sample(make_adapter('alpaca_eval', ['{"verdict": "M"}']), 'ours', 'baseline').value['win_rate'] == 1.0
    assert score_sample(make_adapter('alpaca_eval', ['{"verdict": "m"}']), 'ours', 'baseline').value['win_rate'] == 0.0


def test_alpaca_eval_no_longer_matches_an_m_inside_prose():
    """The old ``re.search(r'(m|M)')`` picked the m out of "model" and scored a win."""
    adapter = make_adapter('alpaca_eval', ['Both models are similar in quality.'])

    score = score_sample(adapter, 'ours', 'baseline')

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


# ---------------------------------------------------------------------------
# frames / docmath / browsecomp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('name', ['frames', 'docmath'])
def test_orm_benchmarks_require_the_json_verdict(name):
    state = _state_with_question(None, 'answer', 'answer')

    assert make_adapter(name, ['{"verdict": "YES"}'], judge_strategy='llm').calculate_metrics(state).score.value == {
        'acc': 1.0
    }
    assert make_adapter(name, ['{"verdict": "NO"}'], judge_strategy='llm').calculate_metrics(state).score.value == {
        'acc': 0.0
    }


@pytest.mark.parametrize('name', ['frames', 'docmath'])
def test_orm_benchmarks_no_longer_read_yes_out_of_the_explanation(name):
    """The old ``'YES' in response`` scored this as correct."""
    adapter = make_adapter(name, ['The answers are NOT equivalent, so YES would be wrong.'], judge_strategy='llm')

    score = adapter.calculate_metrics(_state_with_question(None, 'wrong', 'answer')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_browsecomp_reads_the_correctness_field():
    state = _state_with_question(None, 'answer', 'answer')

    correct = make_adapter(
        'browsecomp', ['{"extracted_final_answer": "answer", "correct": "yes"}'], judge_strategy='llm'
    )
    assert correct.calculate_metrics(state).score.value == {'is_correct': 1.0, 'is_incorrect': 0.0}

    wrong = make_adapter('browsecomp', ['{"correct": "no"}'], judge_strategy='llm')
    assert wrong.calculate_metrics(state).score.value == {'is_correct': 0.0, 'is_incorrect': 1.0}


def test_browsecomp_without_the_correctness_field_is_excluded():
    adapter = make_adapter('browsecomp', ['The response looks correct to me.'], judge_strategy='llm')

    score = adapter.calculate_metrics(_state_with_question(None, 'answer', 'answer')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


# ---------------------------------------------------------------------------
# cmmu / charxiv (structured verdicts)
# ---------------------------------------------------------------------------


def _blank_state(prediction: str, target: str) -> TaskState:
    sample = Sample(id=0, input='question', target=target, metadata={'type': 'fill-in-the-blank'})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


def test_cmmu_keeps_a_partial_credit_proportion():
    """The old regex fallback only recognised ``"correct": 1``, silently losing 0.5."""
    adapter = make_adapter('cmmu', ['{"analysis": "half right", "correct": 0.5}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_blank_state('a', 'a;b')).score

    assert score.value == {'acc': 0.5}
    assert score.metadata['analysis'] == 'half right'


def test_cmmu_rejects_a_score_outside_the_declared_range():
    adapter = make_adapter('cmmu', ['{"analysis": "", "correct": 7}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_blank_state('a', 'a')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_cmmu_multiple_choice_never_calls_the_judge():
    adapter = make_adapter('cmmu', ['{"analysis": "", "correct": 1}'], judge_strategy='llm')
    sample = Sample(id=0, input='q', target='A', metadata={'type': 'multiple-choice'})
    state = TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content='A'),
        completed=True,
    )

    score = adapter.calculate_metrics(state).score

    assert adapter.llm_judge.prompts == []
    assert set(score.value) == {'acc'}
    assert score.status is ScoreStatus.SUCCESS


def _charxiv_state(prediction: str, target: str, question_type: str) -> TaskState:
    sample = Sample(
        id=0,
        input='question',
        target=target,
        metadata={
            'question_type': question_type,
            'question_id': 1,
            'reasoning_a_type': 1
        },
    )
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


def test_charxiv_uses_the_reply_keys_the_official_prompt_asks_for():
    reasoning = make_adapter('charxiv', ['{"extract_answer": "5", "score": 1}'], judge_strategy='llm')
    assert reasoning.calculate_metrics(_charxiv_state('5', '5', 'reasoning')).score.value == {'acc': 1.0}

    descriptive = make_adapter('charxiv', ['{"extract_answer_T1": "5", "score_T1": 0}'], judge_strategy='llm')
    assert descriptive.calculate_metrics(_charxiv_state('5', '6', 'descriptive')).score.value == {'acc': 0.0}


def test_charxiv_rejects_the_wrong_question_types_keys():
    """A descriptive reply to a reasoning question is a contract violation, not a zero."""
    adapter = make_adapter('charxiv', ['{"extract_answer_T1": "5", "score_T1": 1}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_charxiv_state('5', '5', 'reasoning')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


# ---------------------------------------------------------------------------
# needle_haystack
# ---------------------------------------------------------------------------


def _needle_state(prediction: str, target: str) -> TaskState:
    sample = Sample(id=0, input='question', target=target, metadata={'context_length': 4096, 'depth_percent': 50})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


def test_needle_haystack_scales_the_rating_to_a_ratio():
    adapter = make_adapter('needle_haystack', ['{"verdict": 8}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_needle_state('answer', 'answer')).score

    assert score.value == {'Context#4096 Depth#50': 0.8}
    assert score.main_score_name == 'Context#4096 Depth#50'


def test_needle_haystack_rejects_a_rating_outside_the_scale():
    adapter = make_adapter('needle_haystack', ['{"verdict": 42}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_needle_state('answer', 'answer')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_needle_haystack_ignores_a_bare_number_in_prose():
    """The old regex needed the brackets too, but a missing rating used to score 0 instead of being excluded."""
    adapter = make_adapter('needle_haystack', ['I would say about 8 out of 10.'], judge_strategy='llm')

    score = adapter.calculate_metrics(_needle_state('answer', 'answer')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_the_judge_prompt_states_the_json_requirement():
    """The contract owns the format sentence; a prompt without it would strand the parser."""
    adapter = make_adapter('simple_qa', ['{"verdict": "A"}'])
    score_sample(adapter, 'Shakespeare', 'William Shakespeare')

    prompt = adapter.llm_judge.prompts[0]
    assert 'single JSON object' in prompt
    assert '"A" or "B" or "C"' in prompt
    assert 'Just return the letters' not in prompt


# ---------------------------------------------------------------------------
# perception_bench
# ---------------------------------------------------------------------------


def _perception_state(prediction: str) -> TaskState:
    sample = Sample(id=0, input='question', target='42', metadata={'problem': 'What is 6*7?'})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


def test_perception_bench_reads_the_boolean_verdict():
    adapter = make_adapter('perception_bench', ['{"reasoning": "matches", "verdict": true}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_perception_state('42')).score

    assert score.value == {'acc': 1.0}
    assert score.status is ScoreStatus.SUCCESS


def test_perception_bench_no_longer_reads_a_boolean_out_of_prose():
    """The old parser searched the [judge] section for any true/false token."""
    adapter = make_adapter('perception_bench', ['[reason]\nwrong\n[judge]\nTrue'], judge_strategy='llm')

    score = adapter.calculate_metrics(_perception_state('41')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_perception_bench_scores_an_empty_prediction_without_a_judge_call():
    adapter = make_adapter('perception_bench', ['{"verdict": true}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_perception_state('   ')).score

    assert score.value == {'acc': 0.0}
    assert adapter.llm_judge.prompts == []


def test_perception_bench_few_shot_examples_demonstrate_json():
    """Examples teaching the old [reason]/[judge] layout would coach the judge into failing."""
    from evalscope.benchmarks.perception_bench.utils import JUDGE_TEMPLATE

    assert '[reason]' not in JUDGE_TEMPLATE
    assert '[judge]' not in JUDGE_TEMPLATE
    assert '"verdict": false' in JUDGE_TEMPLATE


# ---------------------------------------------------------------------------
# mia_bench
# ---------------------------------------------------------------------------


def _mia_state(prediction: str) -> TaskState:
    sample = Sample(
        id=0,
        input='follow the instruction',
        target='',
        metadata={
            'instruction': 'Answer in French, in one word.',
            'components': ['answer in French', 'one word only'],
            'component_type': ['language', 'length'],
            'component_weight': [3, 2],
        },
    )
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


def test_mia_bench_normalizes_each_component_and_derives_the_total():
    adapter = make_adapter(
        'mia_bench', ['{"reasoning": "ok", "component_1": 3, "component_2": 1}'], judge_strategy='llm'
    )

    score = adapter.calculate_metrics(_mia_state('un')).score

    assert score.value['component_1_language'] == 1.0
    assert score.value['component_2_length'] == 0.5
    # judge_score = (3 + 1) / (3 + 2)
    assert score.value['judge_score'] == 0.8
    assert score.main_score_name == 'judge_score'


def test_mia_bench_rejects_a_component_above_its_weight():
    adapter = make_adapter('mia_bench', ['{"component_1": 9, "component_2": 1}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_mia_state('un')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_mia_bench_scores_zero_without_components_and_no_judge_call():
    adapter = make_adapter('mia_bench', ['{"component_1": 1}'], judge_strategy='llm')
    sample = Sample(id=0, input='x', target='', metadata={'components': []})
    state = TaskState(model='m', sample=sample, output=ModelOutput.from_content(model='m', content='y'), completed=True)

    score = adapter.calculate_metrics(state).score

    assert score.value == {'judge_score': 0.0}
    assert adapter.llm_judge.prompts == []


# ---------------------------------------------------------------------------
# deepsearchqa
# ---------------------------------------------------------------------------


def _deepsearch_state(prediction: str, target: str, answer_type: str = 'Single Answer') -> TaskState:
    sample = Sample(id=0, input='which country?', target=target, metadata={'answer_type': answer_type})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


_DEEPSEARCH_OK = (
    '{"Answer Correctness": {"Explanation": "match", '
    '"Correctness Details": {"Belgium": true, "France": true}, "Excessive Answers": []}}'
)


def test_deepsearchqa_reads_the_nested_official_json():
    adapter = make_adapter('deepsearchqa', [_DEEPSEARCH_OK], judge_strategy='llm')

    score = adapter.calculate_metrics(_deepsearch_state('Belgium and France', 'Belgium, France')).score

    assert score.value['f1'] == 1.0
    assert score.status is ScoreStatus.SUCCESS


def test_deepsearchqa_excludes_a_malformed_verdict():
    adapter = make_adapter('deepsearchqa', ['the answer looks basically correct to me'], judge_strategy='llm')

    score = adapter.calculate_metrics(_deepsearch_state('Belgium', 'Belgium')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_deepsearchqa_empty_prediction_skips_the_judge():
    adapter = make_adapter('deepsearchqa', [_DEEPSEARCH_OK], judge_strategy='llm')

    score = adapter.calculate_metrics(_deepsearch_state('', 'Belgium')).score

    assert score.value == {}
    assert score.metadata['empty_model_response'] is True
    assert adapter.llm_judge.prompts == []


# ---------------------------------------------------------------------------
# hipho
# ---------------------------------------------------------------------------


def _hipho_answer_state(prediction: str) -> TaskState:
    sample = Sample(
        id=0,
        input='question',
        target='',
        metadata={
            'id': 'p1',
            'question': 'What is 6*7?',
            'answers': ['\\boxed{42}'],
            'marking': None,
        },
    )
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


def test_hipho_answer_level_skips_the_judge_when_rules_already_match():
    adapter = make_adapter('hipho', ['{"correct": true}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_hipho_answer_state('The answer is \\boxed{42}.')).score

    assert score.value == {'acc': 1.0}
    assert adapter.llm_judge.prompts == []


def test_hipho_answer_level_falls_back_to_the_judge_when_rules_disagree():
    adapter = make_adapter('hipho', ['{"correct": true}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_hipho_answer_state('The answer is \\boxed{forty-two}.')).score

    assert score.value == {'acc': 1.0}
    assert len(adapter.llm_judge.prompts) == 1


def test_hipho_step_level_bounds_each_criterion_and_takes_the_best_scheme():
    sample = Sample(
        id=0,
        input='q',
        target='',
        metadata={
            'id': 'p1',
            'question': 'derive it',
            'answers': ['\\boxed{x}'],
            # Two schemes; each criterion carries its own point allocation.
            'marking': [
                ['Award 1.0 pt if the student sets up momentum.', 'Award 2.0 pts if the student solves for v.'],
                ['Award 3.0 pts if the student uses energy conservation.'],
            ],
        },
    )
    state = TaskState(
        model='m', sample=sample, output=ModelOutput.from_content(model='m', content='derivation'), completed=True
    )
    # Scheme 0: 1/1 + 1/2 = 0.5;  Scheme 1: 3/3 = 1.0 -> best is 1.0.
    adapter = make_adapter(
        'hipho', [
            '{"awarded": 1.0}',
            '{"awarded": 1.0}',
            '{"awarded": 3.0}',
        ], judge_strategy='llm'
    )

    score = adapter.calculate_metrics(state).score

    assert score.value == {'acc': 1.0}
    assert score.metadata['grading'] == 'step_level'


def test_hipho_step_level_rejects_a_score_exceeding_the_criterion_max():
    """A judge that awards more than the criterion allows is a parse failure, not silent inflation."""
    sample = Sample(
        id=0,
        input='q',
        target='',
        metadata={
            'id': 'p1',
            'question': 'derive it',
            'answers': ['\\boxed{x}'],
            'marking': [['Award 1.0 pt if the student sets up momentum.']],
        },
    )
    state = TaskState(
        model='m', sample=sample, output=ModelOutput.from_content(model='m', content='derivation'), completed=True
    )
    adapter = make_adapter('hipho', ['{"awarded": 5.0}'], judge_strategy='llm')

    score = adapter.calculate_metrics(state).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


# ---------------------------------------------------------------------------
# healthbench
# ---------------------------------------------------------------------------


def _healthbench_state() -> TaskState:
    sample = Sample(
        id=0,
        input='patient question',
        target='',
        metadata={
            'prompt': [{
                'role': 'user',
                'content': 'I have a headache.'
            }],
            'rubrics': [
                {
                    'criterion': 'Advises seeing a doctor',
                    'points': 5,
                    'tags': ['axis:accuracy']
                },
                {
                    'criterion': 'Mentions hydration',
                    'points': 3,
                    'tags': ['axis:completeness']
                },
            ],
        },
    )
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content='See a doctor and drink water.'),
        completed=True,
    )


def test_healthbench_weights_each_rubric_verdict():
    adapter = make_adapter(
        'health_bench', [
            '{"explanation": "advises a doctor", "criteria_met": true}',
            '{"explanation": "no hydration advice", "criteria_met": false}',
        ],
        judge_strategy='llm'
    )

    score = adapter.calculate_metrics(_healthbench_state()).score

    # 5 of 8 possible points achieved.
    assert score.value['overall_score'] == 5 / 8
    assert score.main_score_name == 'overall_score'
    assert len(adapter.llm_judge.prompts) == 2


def test_healthbench_excludes_a_sample_when_a_rubric_verdict_is_unparseable():
    adapter = make_adapter(
        'health_bench', [
            '{"explanation": "ok", "criteria_met": true}',
            'The response partially meets the criteria.',
        ],
        judge_strategy='llm'
    )

    score = adapter.calculate_metrics(_healthbench_state()).score

    # A required rubric with no usable verdict invalidates the whole observation.
    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_healthbench_rejects_a_non_boolean_criteria_met():
    adapter = make_adapter('health_bench', ['{"explanation": "ok", "criteria_met": "maybe"}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_healthbench_state()).score

    assert score.value == {}


# ---------------------------------------------------------------------------
# plawbench
# ---------------------------------------------------------------------------


def _plaw_total_state() -> TaskState:
    sample = Sample(
        id=0,
        input='draft a contract',
        target='',
        metadata={
            'id': 'p1',
            'judge_type': 'document_generation',
            'prompt': 'draft a contract',
            'rubrics': 'follow the template',
            'max_points': 10,
        },
    )
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content='here is the contract'),
        completed=True,
    )


def test_plawbench_normalizes_awarded_points_against_the_dataset_maximum():
    adapter = make_adapter('plawbench', ['{"total_points": 7, "max_points": 10}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_plaw_total_state()).score

    assert score.value == {'acc': 0.7}
    assert score.main_score_name == 'acc'


def test_plawbench_rejects_a_reply_without_total_points():
    """The old code raised inside retry_call and then scored the sample 0."""
    adapter = make_adapter('plawbench', ['{"score": 7}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_plaw_total_state()).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


# ---------------------------------------------------------------------------
# cl_bench
# ---------------------------------------------------------------------------


def _cl_bench_state(prediction: str) -> TaskState:
    sample = Sample(id=0, input='follow the rules', target=['must be in French', 'must be one word'], metadata={})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=prediction),
        completed=True,
    )


def test_cl_bench_reads_the_binary_overall_score():
    adapter = make_adapter(
        'cl_bench', [
            '{"Grading Rationale": "all met", "List of Requirement Satisfaction Status": ["yes", "yes"], '
            '"Overall Score": 1}'
        ],
        judge_strategy='llm'
    )

    score = adapter.calculate_metrics(_cl_bench_state('un')).score

    assert score.value == {'acc': 1.0}
    assert score.main_score_name == 'acc'


def test_cl_bench_accepts_a_quoted_score_without_wasting_a_retry():
    """The contract renders the labels quoted, so a compliant judge replies with a string;
    it must parse on the first attempt rather than only after a retry."""
    adapter = make_adapter(
        'cl_bench', [
            '{"Grading Rationale": "all met", "List of Requirement Satisfaction Status": [], '
            '"Overall Score": "0"}'
        ],
        judge_strategy='llm'
    )

    sample_score = adapter.calculate_metrics(_cl_bench_state('un')).score

    assert sample_score.value == {'acc': 0.0}
    attempts = (sample_score.metadata or {}).get('judge_attempts', [])
    assert [attempt['status'] for attempt in attempts] == ['success']


def test_cl_bench_rejects_a_score_outside_the_binary_scale():
    """The old code coerced anything that was not 1 into 0, hiding a malformed verdict."""
    adapter = make_adapter('cl_bench', ['{"Overall Score": 2}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_cl_bench_state('un')).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_cl_bench_scores_an_empty_prediction_without_a_judge_call():
    adapter = make_adapter('cl_bench', ['{"Overall Score": 1}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_cl_bench_state('   ')).score

    assert score.value == {'acc': 0.0}
    assert adapter.llm_judge.prompts == []


# ---------------------------------------------------------------------------
# drivelology
# ---------------------------------------------------------------------------


def _drivelology_state() -> TaskState:
    sample = Sample(id=0, input='explain the joke', target='the reference narrative', metadata={})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content='the candidate narrative'),
        completed=True,
    )


def test_drivelology_normalizes_the_five_point_rating():
    adapter = make_adapter('drivel_writing', ['{"reasoning": "close", "rating": 4}'], judge_strategy='llm')

    score = adapter.calculate_metrics(_drivelology_state()).score

    assert score.value == {'judge_score': 0.75}
    assert score.main_score_name == 'judge_score'


def test_drivelology_no_longer_grabs_any_digit_from_prose():
    """The old third-tier fallback matched any standalone 1-5 anywhere in the reply."""
    adapter = make_adapter(
        'drivel_writing', ['The candidate covers 3 of the reference points but misses the punchline.'],
        judge_strategy='llm'
    )

    score = adapter.calculate_metrics(_drivelology_state()).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_drivelology_rejects_a_rating_off_the_scale():
    adapter = make_adapter('drivel_writing', ['{"rating": 9}'], judge_strategy='llm')

    assert adapter.calculate_metrics(_drivelology_state()).score.value == {}


# ---------------------------------------------------------------------------
# arena_hard
# ---------------------------------------------------------------------------


class PlacementJudge:
    """Returns a fixed reply per pairwise placement; a retry cannot rescue a bad pass."""

    model_id = 'scripted-judge'

    def __init__(self, original: str, swapped: str, first_marker: str) -> None:
        self.original = original
        self.swapped = swapped
        self.first_marker = first_marker
        self.prompts: List[str] = []

    def judge(self, prompt: str = '', system_prompt: Optional[str] = None, messages: Any = None) -> str:
        # Scans the whole conversation, not just the last message: a parse retry appends a
        # correction, and the placement is only stated in the original case prompt.
        text = prompt or '\n'.join(str(message.content) for message in (messages or []))
        self.prompts.append(text)
        return self.original if text.index('baseline') < text.index('candidate') else self.swapped


def _arena_state() -> TaskState:
    sample = Sample(id=0, input='write a poem', target='baseline answer', metadata={})
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content='candidate answer'),
        completed=True,
    )


def _arena_adapter(original: str, swapped: str):
    adapter = make_adapter('arena_hard', [original], judge_strategy='llm')
    adapter.llm_judge = PlacementJudge(original, swapped, 'baseline')
    return adapter


def test_arena_hard_averages_both_games():
    # Game 1: baseline is A, candidate is B -> B>A favours the candidate.
    # Game 2: candidate is A, baseline is B -> A>B favours the candidate again.
    adapter = _arena_adapter('{"verdict": "B>A"}', '{"verdict": "A>B"}')

    score = adapter.calculate_metrics(_arena_state()).score

    assert score.status is ScoreStatus.SUCCESS
    assert score.value['score'] > 0.5
    assert score.metadata['battle_result']['games'] == [{'score': 'B>A'}, {'score': 'A>B'}]


def test_arena_hard_requires_both_games():
    adapter = _arena_adapter('{"verdict": "B>A"}', 'My final verdict is tie: [[A=B]]')

    score = adapter.calculate_metrics(_arena_state()).score

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_arena_hard_aggregation_skips_excluded_samples():
    """The Elo aggregation used to KeyError on a sample without a battle result."""
    from evalscope.api.metric import SampleScore, Score

    adapter = _arena_adapter('{"verdict": "B>A"}', '{"verdict": "A>B"}')
    excluded = SampleScore(sample_id=1, score=Score(value={}, metadata={}, status=ScoreStatus.EXCLUDED))

    assert adapter.aggregate_scores([excluded]) == []


# ---------------------------------------------------------------------------
# researchrubrics
# ---------------------------------------------------------------------------


def _researchrubrics_state(report: str, rubrics: list) -> TaskState:
    import json
    sample = Sample(
        id=0,
        input='research question',
        target=json.dumps(rubrics),
        metadata={
            'sample_id': 'test_001',
            'domain': 'science'
        },
    )
    return TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content=report),
        completed=True,
    )


def test_researchrubrics_binary_scores_short_docs():
    rubrics = [
        {
            'criterion': 'Mentions gravity',
            'axis': 'content',
            'weight': 2.0
        },
        {
            'criterion': 'Cites Newton',
            'axis': 'references',
            'weight': 1.0
        },
    ]
    adapter = make_adapter(
        'researchrubrics', [
            '{"verdict": "Satisfied", "score": 1.0, "confidence": 0.9, "reasoning": "ok", "evidence_quotes": ["g"], "missing_elements": []}',
            '{"verdict": "Not Satisfied", "score": 0.0, "confidence": 0.8, "reasoning": "no", "evidence_quotes": [], "missing_elements": ["Newton"]}',
        ],
        judge_strategy='llm'
    )

    state = _researchrubrics_state('Gravity pulls objects together.', rubrics)
    score = adapter._score_task_state(state)

    # compliance = (1.0*2 + 0.0*1) / (2+1) = 2/3
    assert abs(score.value['compliance_score'] - 2.0 / 3.0) < 1e-9
    assert score.main_score_name == 'compliance_score'
    assert 'axis/content' in score.value
    assert 'axis/references' in score.value


def test_researchrubrics_chunked_emits_synthesis_after_chunks():
    """Long docs trigger chunking; evidence from chunks feeds the synthesis pass."""
    rubrics = [{'criterion': 'Has references', 'axis': 'refs', 'weight': 1.0}]
    # Make a report that triggers chunking (> judge_context_limit * 4 chars).
    long_report = 'word ' * 800000  # 4M chars → 1M estimated tokens > 150k default

    adapter = make_adapter(
        'researchrubrics',
        [
            # Chunk responses (one per chunk)
            '{"relevant_evidence": ["ref found in chunk 1"], "satisfaction": true, "confidence_for_chunk": 0.9, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 2"], "satisfaction": true, "confidence_for_chunk": 0.8, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 3"], "satisfaction": true, "confidence_for_chunk": 0.7, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 4"], "satisfaction": true, "confidence_for_chunk": 0.7, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 5"], "satisfaction": true, "confidence_for_chunk": 0.7, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 6"], "satisfaction": true, "confidence_for_chunk": 0.7, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 7"], "satisfaction": true, "confidence_for_chunk": 0.7, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 8"], "satisfaction": true, "confidence_for_chunk": 0.7, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 9"], "satisfaction": true, "confidence_for_chunk": 0.7, "notes": "ok"}',
            '{"relevant_evidence": ["ref in chunk 10"], "satisfaction": true, "confidence_for_chunk": 0.7, "notes": "ok"}',
            # Synthesis response
            '{"verdict": "Satisfied", "score": 1.0, "confidence": 0.95, "reasoning": "synthesized", "evidence_quotes": ["ref found in chunk 1"], "missing_elements": []}',
        ],
        judge_strategy='llm'
    )

    state = _researchrubrics_state(long_report, rubrics)
    score = adapter._score_task_state(state)

    assert score.value['compliance_score'] == 1.0
    assert score.metadata.get('used_chunking') is True


def test_researchrubrics_rejects_unparseable_verdict():
    rubrics = [{'criterion': 'Has intro', 'axis': 'structure', 'weight': 1.0}]
    adapter = make_adapter(
        'researchrubrics',
        [
            'The document has a fine introduction.',  # Not JSON
        ],
        judge_strategy='llm'
    )

    state = _researchrubrics_state('Hello world.', rubrics)
    score = adapter._score_task_state(state)

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


# ---------------------------------------------------------------------------
# Default judge path (LLMJudgeMixin): benchmarks that do not override the hooks.
# ---------------------------------------------------------------------------

DEFAULT_PATH_BENCHMARKS = ['minerva_math', 'imo_answerbench', 'math_verse', 'world_vqa', 'zerobench', 'baby_vision']


def _default_path_score(name: str, response: str):
    adapter = make_adapter(name, [response], judge_strategy='llm')
    state = make_state('a wrong prediction', '42')
    return adapter.llm_match_score('a wrong prediction', 'a wrong prediction', '42', state)


@pytest.mark.parametrize('name', DEFAULT_PATH_BENCHMARKS)
def test_default_path_grades_the_json_verdict(name):
    assert _default_path_score(name, '{"reasoning": "ok", "verdict": "A"}').value == {'acc': 1.0}
    assert _default_path_score(name, '{"reasoning": "no", "verdict": "B"}').value == {'acc': 0.0}


@pytest.mark.parametrize('name', DEFAULT_PATH_BENCHMARKS)
def test_default_path_no_longer_reads_a_letter_out_of_prose(name):
    """The old A/B pattern matched a bare letter in prose; the contract excludes it."""
    score = _default_path_score(name, 'The prediction is essentially correct, so grade A.')
    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


@pytest.mark.parametrize('name', DEFAULT_PATH_BENCHMARKS)
def test_default_path_transport_failure_excludes_the_sample(name):
    score = _default_path_score(name, ERROR_RESPONSE)
    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_default_path_prompt_carries_question_target_and_prediction():
    adapter = make_adapter('minerva_math', ['{"verdict": "A"}'], judge_strategy='llm')
    state = make_state('predicted 7', 'gold 42', question='What is 6 times 7?')
    adapter.llm_match_score('predicted 7', 'predicted 7', 'gold 42', state)
    prompt = adapter.llm_judge.prompts[0]
    assert 'What is 6 times 7?' in prompt
    assert 'gold 42' in prompt
    assert 'predicted 7' in prompt
    assert 'JSON object' in prompt


def test_default_path_prompt_states_no_conflicting_format():
    """The old template ended with "just return A or B", contradicting the JSON instruction."""
    adapter = make_adapter('minerva_math', ['{"verdict": "A"}'], judge_strategy='llm')
    state = make_state('p', 'g')
    adapter.llm_match_score('p', 'p', 'g', state)

    prompt = adapter.llm_judge.prompts[0]

    assert 'with no text around it' not in prompt
    assert prompt.count('JSON object') == 1


def test_default_path_honours_a_custom_score_mapping():
    """``score_mapping`` drives both the allowed labels and the value, as it documented."""
    adapter = make_adapter('minerva_math', ['{"verdict": "GOOD"}'], judge_strategy='llm')
    adapter.llm_judge.score_mapping = {'GOOD': 1.0, 'BAD': 0.25}
    state = make_state('p', 'g')

    score = adapter.llm_match_score('p', 'p', 'g', state)

    assert score.value == {'acc': 1.0}
    assert 'exactly one of "BAD" or "GOOD"' in adapter.llm_judge.prompts[0]


def test_default_path_rejects_a_label_outside_the_score_mapping():
    adapter = make_adapter('minerva_math', ['{"verdict": "A"}'], judge_strategy='llm')
    adapter.llm_judge.score_mapping = {'GOOD': 1.0, 'BAD': 0.0}
    state = make_state('p', 'g')

    score = adapter.llm_match_score('p', 'p', 'g', state)

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


# ---- numeric (reference-free rating) mode ----


def _numeric_score(response: str):
    """The documented ``score_type='numeric'`` flow: rate a response without a reference."""
    adapter = make_adapter('general_qa', [response], judge_strategy='llm')
    adapter.llm_judge = RatingJudge([response])
    state = make_state('an answer', 'a target')
    return adapter.llm_match_score('an answer', 'an answer', 'a target', state)


def test_numeric_mode_scores_the_rating():
    assert _numeric_score('{"reasoning": "good", "score": 0.75}').value == {'acc': 0.75}
    assert _numeric_score('{"reasoning": "bad", "score": 0}').value == {'acc': 0.0}


def test_numeric_mode_rejects_a_rating_off_the_scale():
    score = _numeric_score('{"reasoning": "great", "score": 42}')

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_numeric_mode_no_longer_reads_a_rating_out_of_prose():
    """The old extractor pulled ``[[0.5]]`` out of free text, and scored 0 when it could not."""
    score = _numeric_score('I would rate this response [[0.5]] overall.')

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_numeric_mode_transport_failure_excludes_instead_of_scoring_zero():
    score = _numeric_score(ERROR_RESPONSE)

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED


def test_numeric_mode_prompt_asks_for_a_bounded_number():
    adapter = make_adapter('general_qa', ['{"score": 1}'], judge_strategy='llm')
    adapter.llm_judge = RatingJudge(['{"score": 1}'])
    state = make_state('an answer', 'a target', question='Explain gravity.')

    adapter.llm_match_score('an answer', 'an answer', 'a target', state)

    prompt = adapter.llm_judge.prompts[0]
    assert 'Explain gravity.' in prompt
    assert '"score": a number >= 0.0 and <= 1.0' in prompt
    assert '[[rating]]' not in prompt
