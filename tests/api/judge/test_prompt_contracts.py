"""Judge prompt examples and rendered requests must agree with the JSON contract."""

import json

import pytest

from evalscope.api.judge import OutputContract
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.cmmu.cmmu_adapter import GRADE_CONTRACT as CMMU_CONTRACT
from evalscope.benchmarks.cmmu.prompt import EVALUATION_SYSTEM_PROMPT
from evalscope.benchmarks.deepsearchqa.utils import GRADE_CONTRACT as DEEPSEARCH_CONTRACT
from evalscope.benchmarks.deepsearchqa.utils import build_grader_prompt
from evalscope.benchmarks.researchrubrics.utils import (
    BINARY_CONTRACT,
    BINARY_USER_PROMPT,
    CHUNK_CONTRACT,
    CHUNK_USER_PROMPT,
    SYNTHESIS_USER_PROMPT,
)
from evalscope.benchmarks.wide_search.utils import EVAL_COLUMN_PROMPT
from evalscope.benchmarks.wide_search.wide_search_adapter import _column_score_model
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus
from tests.api.judge.helpers import ScriptedJudge, make_state


def test_deepsearch_prompt_example_is_a_complete_schema_valid_object() -> None:
    prompt = build_grader_prompt('Name two countries.', 'Belgium, France', 'Set Answer', 'Belgium, France, Italy')
    example = prompt.split('**Example (Partial):**')[1].split('**Now, proceed')[0].strip()

    assert DEEPSEARCH_CONTRACT.parse(example).ok
    assert '```' not in prompt
    assert 'return NULL' not in prompt


@pytest.mark.parametrize('template', [BINARY_USER_PROMPT, SYNTHESIS_USER_PROMPT, CHUNK_USER_PROMPT])
def test_researchrubrics_examples_are_valid_against_their_declared_schema(template: str) -> None:
    prompt = template.format(
        document_content='Report', rubric_title='Criterion', rubric_category='Other', rubric_weight=1,
        chunk_num=1, total_chunks=1, context_summary='', chunk_content='Report', all_evidence='Evidence',
    )
    example = prompt.split('Example:\n')[1].split('Ensure your response')[0].strip()
    contract = CHUNK_CONTRACT if template == CHUNK_USER_PROMPT else BINARY_CONTRACT

    assert contract.parse(example).ok
    assert '```' not in prompt


def test_cmmu_prompt_example_is_valid_against_its_schema() -> None:
    prompt = EVALUATION_SYSTEM_PROMPT
    example = prompt[prompt.index('{'):prompt.rindex('}') + 1]

    assert CMMU_CONTRACT.parse(example).ok


def test_wide_search_column_prompt_example_is_valid_against_its_schema() -> None:
    prompt = EVAL_COLUMN_PROMPT.format(criterion='Same meaning.', response='Rows to grade.')
    example = prompt.split('as shown below:\n')[1].split('====== criterion-start')[0].strip()
    contract = OutputContract(schema_model=_column_score_model(2))

    assert contract.parse(example).ok
    assert '```' not in prompt


@pytest.mark.parametrize('judge_type', ['legal_qa', 'document_generation', 'case_analysis'])
@pytest.mark.parametrize('envelope', ['{reply}', '```json\n{reply}\n```', 'Analysis first.\n{reply}'])
def test_plawbench_prompts_and_reply_contract(judge_type: str, envelope: str) -> None:
    config = TaskConfig(model='m', datasets=['plawbench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark('plawbench', config)
    payload = {'reasoning': 'The rubric is satisfied.', 'total_points': 5, 'max_points': 10}
    if judge_type == 'case_analysis':
        payload = {'reasoning': 'The rubric is satisfied.', 'score_details': {'结论': {'total_points': 5}}}
    reply = json.dumps(payload, ensure_ascii=False)
    judge = ScriptedJudge([envelope.format(reply=reply)])
    adapter.llm_judge = judge
    state = make_state('Response', '')
    state.metadata = {
        'judge_type': judge_type,
        'prompt': 'Question',
        'rubrics': json.dumps([{'tags': '结论得分', 'points': 10}]),
        'max_points': 10,
    }

    score = adapter.score_with_judge_contracts('Response', 'Response', '', state)

    prompt = '\n'.join(message.text for message in judge.calls[0])
    assert 'reasoning' in prompt and 'single JSON object and no other text' in prompt
    assert '```' not in prompt and '在回答的结尾' not in prompt
    if judge_type == 'case_analysis':
        assert '"score_details": an object' in prompt
    if envelope == '{reply}':
        assert score.status.is_usable
        assert score.value['acc'] == 0.5
    else:
        assert score.status is ScoreStatus.EXCLUDED
        assert score.value == {}
        assert score.judge_summary.failures == {'parse_error': 1}
    assert len(judge.calls) == 1


def test_cl_bench_rationale_is_requested_inside_the_json_object() -> None:
    config = TaskConfig(model='m', datasets=['cl_bench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark('cl_bench', config)
    judge = ScriptedJudge([json.dumps({
        'Grading Rationale': 'All requirements met.',
        'List of Requirement Satisfaction Status': ['met'],
        'Overall Score': 1,
    })])
    adapter.llm_judge = judge

    score = adapter.score_with_judge_contracts('Response', 'Response', 'Rubric', make_state('Response', 'Rubric'))

    assert score.status.is_usable and score.value['acc'] == 1.0
    prompt = '\n'.join(message.text for message in judge.calls[0])
    assert 'JSON Grading Rationale field' in prompt
    assert 'Do not write analysis outside the JSON object' in prompt
    assert 'must be a list of strings, not a list of objects' in prompt


def test_plawbench_missing_dataset_section_is_excluded_before_reduction() -> None:
    config = TaskConfig(model='m', datasets=['plawbench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark('plawbench', config)
    judge = ScriptedJudge([json.dumps({'score_details': {'conclusion': {'total_points': 5}}})])
    adapter.llm_judge = judge
    state = make_state('Response', '')
    state.metadata = {
        'judge_type': 'case_analysis', 'prompt': 'Question',
        'rubrics': json.dumps([{'tags': '结论得分', 'points': 10}]), 'max_points': 10,
    }

    score = adapter.score_with_judge_contracts('Response', 'Response', '', state)

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.judge_summary.failures == {'parse_error': 1}
    assert len(judge.calls) == 1
    prompt = '\n'.join(message.text for message in judge.calls[0])
    assert 'JSON Schema:' in prompt and '结论' in prompt
