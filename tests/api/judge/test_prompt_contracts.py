"""Judge prompt examples and rendered requests must agree with the JSON contract."""

import json
from pathlib import Path

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.judge import OutputContract
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.charxiv.charxiv_adapter import DESCRIPTIVE_CONTRACT
from evalscope.benchmarks.charxiv.utils import DESCRIPTIVE_GRADING_ICL
from evalscope.benchmarks.cmmu.cmmu_adapter import GRADE_CONTRACT as CMMU_CONTRACT
from evalscope.benchmarks.cmmu.prompt import EVALUATION_SYSTEM_PROMPT
from evalscope.benchmarks.deepsearchqa.utils import GRADE_CONTRACT as DEEPSEARCH_CONTRACT
from evalscope.benchmarks.deepsearchqa.utils import build_grader_prompt
from evalscope.benchmarks.image_edit.gedit import vie_prompts
from evalscope.benchmarks.image_edit.gedit.gedit_adapter import GEDIT_CONTRACT
from evalscope.benchmarks.job_bench.job_bench_adapter import _rubric_verdict_model
from evalscope.benchmarks.job_bench.utils import build_judge_prompt
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
from evalscope.evaluation_versioning import SHARED_SCORING_VERSION
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


@pytest.mark.parametrize(('benchmark_name', 'legacy_instruction', 'json_instruction'), [
    ('simple_qa', 'Simply reply with either', 'Set the JSON `verdict` field'),
    ('chinese_simpleqa', '请只回复A、B、C之一', '将所选标签填入 JSON 的 `verdict` 字段'),
    ('simple_vqa', 'Simply reply with either', 'Set the JSON `verdict` field'),
])
def test_three_way_qa_requests_require_a_json_verdict(
    benchmark_name: str, legacy_instruction: str, json_instruction: str,
) -> None:
    config = TaskConfig(model='m', datasets=[benchmark_name], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark(benchmark_name, config)
    judge = ScriptedJudge(['{"verdict": "A"}'])
    adapter.llm_judge = judge

    score = adapter.calculate_metrics(make_state('Shakespeare', 'Shakespeare')).score

    assert score.status.is_usable
    prompt = '\n'.join(message.text for message in judge.calls[0])
    assert legacy_instruction not in prompt
    assert json_instruction in prompt
    assert 'single JSON object and no other text' in prompt


@pytest.mark.parametrize('category', list(DESCRIPTIVE_GRADING_ICL))
def test_charxiv_descriptive_examples_are_json_and_match_the_schema(category: str) -> None:
    rubric = DESCRIPTIVE_GRADING_ICL[category]
    example = rubric.split('### Example Start ###')[1].split('### Example End ###')[0]
    example = example[example.index('{'):example.rindex('}') + 1]
    payload = json.loads(example)

    for suffix in ('T1', 'T2', 'T3'):
        reply = json.dumps({
            'extract_answer_T1': payload[f'extract_answer_{suffix}'],
            'score_T1': payload[f'score_{suffix}'],
        })
        result = DESCRIPTIVE_CONTRACT.parse(reply)
        assert result.ok, f'{category} {suffix}: {result.error}'


@pytest.mark.parametrize('question_type', ['descriptive', 'reasoning'])
@pytest.mark.parametrize('reply_kind', ['valid', 'fenced', 'numeric_answer'])
def test_charxiv_requests_declare_and_enforce_the_final_output_contract(
    question_type: str, reply_kind: str,
) -> None:
    adapter = get_benchmark('charxiv', TaskConfig(
        model='m', datasets=['charxiv'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
    ))
    descriptive = question_type == 'descriptive'
    answer_key, score_key = ('extract_answer_T1', 'score_T1') if descriptive else ('extract_answer', 'score')
    reply = json.dumps({answer_key: 5 if reply_kind == 'numeric_answer' else '5', score_key: 1})
    if reply_kind == 'fenced':
        reply = f'```json\n{reply}\n```'
    judge = ScriptedJudge([reply])
    adapter.llm_judge = judge
    state = make_state('5', '5')
    state.metadata = {'question_type': question_type, 'question_id': 10, 'reasoning_a_type': 3}

    score = adapter.calculate_metrics(state).score

    prompt = judge.calls[0][0].text
    assert 'single JSON object and no other text' in prompt
    assert f'"{answer_key}": a string' in prompt
    assert len(judge.calls) == 1
    if reply_kind == 'valid':
        assert score.status is ScoreStatus.SUCCESS
        assert score.value == {'acc': 1.0}
    else:
        assert score.status is ScoreStatus.EXCLUDED
        assert score.value == {}
        assert score.judge_summary.failures == {'parse_error': 1}


@pytest.mark.parametrize('criteria_count', [0, 1, 2, 3])
def test_job_bench_output_example_matches_the_declared_criteria_count(criteria_count: int) -> None:
    criteria = [f'criterion {index}' for index in range(criteria_count)]
    prompt = build_judge_prompt({'rubric': '', 'criterion': criteria}, 'artifact', False)
    example = prompt.split('## Output Format\n')[1].split('\n\nIMPORTANT:')[0]
    example = example[example.index('{'):example.rindex('}') + 1]

    result = OutputContract(schema_model=_rubric_verdict_model(criteria_count)).parse(example)

    assert result.ok, result.error
    assert len(result.value.criteria_results) == criteria_count
    assert 'true/false' not in example
    assert 'true or false' in prompt.split('IMPORTANT:')[1]
    assert f'exactly {criteria_count} items' in prompt


def test_gedit_output_example_is_json_and_matches_the_schema() -> None:
    prompt = vie_prompts._context_no_delimit
    example = prompt[prompt.index('{'):]

    result = GEDIT_CONTRACT.parse(example)

    assert result.ok, result.error


def _job_bench_state(tmp_path: Path) -> TaskState:
    output_dir = tmp_path / 'jobbench_output'
    output_dir.mkdir()
    (output_dir / 'answer.txt').write_text('artifact', encoding='utf-8')
    state = make_state('artifact', '')
    state.metadata = {
        'artifact_dir': str(tmp_path),
        'rubric_json': [{'rubric': 'Complete answer', 'criterion': ['first', 'second'], 'weight': 2}],
    }
    return state


def test_job_bench_empty_criteria_schema_accepts_an_empty_complete_verdict() -> None:
    contract = OutputContract(schema_model=_rubric_verdict_model(0))

    assert contract.parse('{"rubric_passed":true,"criteria_results":[]}').ok


def test_job_bench_complete_criteria_pass_through_the_executor(tmp_path: Path) -> None:
    adapter = get_benchmark('job_bench', TaskConfig(
        model='m', datasets=['job_bench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
    ))
    adapter.llm_judge = ScriptedJudge([
        '{"rubric_passed":true,"criteria_results":[{"index":0,"passed":true},{"index":1,"passed":true}]}'
    ])

    score = adapter.score_with_judge_contracts('artifact', 'artifact', '', _job_bench_state(tmp_path))

    assert score.status.is_usable
    assert score.value['normalized_score'] == 1.0


@pytest.mark.parametrize('reply', [
    '{"rubric_passed":true}',
    '{"rubric_passed":true,"criteria_results":[{"index":0,"passed":true}]}',
    '{"rubric_passed":true,"criteria_results":['
    '{"index":0,"passed":false},{"index":0,"passed":true},{"index":1,"passed":true}]}',
    '{"rubric_passed":true,"criteria_results":['
    '{"index":0,"passed":true},{"index":1,"passed":true},{"index":99,"passed":false}]}',
])
def test_job_bench_incomplete_or_ambiguous_criteria_are_excluded(reply: str, tmp_path: Path) -> None:
    adapter = get_benchmark('job_bench', TaskConfig(
        model='m', datasets=['job_bench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
    ))
    adapter.llm_judge = ScriptedJudge([reply])

    score = adapter.score_with_judge_contracts('artifact', 'artifact', '', _job_bench_state(tmp_path))

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.judge_summary.failures == {'parse_error': 1}


def _gedit_state() -> TaskState:
    state = make_state('edited', '')
    state.metadata = {'input_image': 'source', 'image_path': 'edited', 'instruction': 'edit'}
    return state


def test_gedit_valid_rating_pair_passes_through_the_executor() -> None:
    adapter = get_benchmark('gedit', TaskConfig(
        model='m', datasets=['gedit'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
    ))
    adapter.llm_judge = ScriptedJudge(['{"score":[5,10]}', '{"score":[5,10]}'])

    score = adapter.score_with_judge_contracts('edited', 'edited', '', _gedit_state())

    assert score.status.is_usable
    assert score.value == {'semantic_consistency': 5.0, 'perceptual_similarity': 5.0, 'normalized_score': 5.0}


@pytest.mark.parametrize('reply', [
    '{"score":[100,100]}',
    '{"score":[-1,-1]}',
    '{"score":[10]}',
    '{"score":[10,10,0]}',
])
def test_gedit_invalid_rating_pairs_are_excluded(reply: str) -> None:
    adapter = get_benchmark('gedit', TaskConfig(
        model='m', datasets=['gedit'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
    ))
    adapter.llm_judge = ScriptedJudge([reply, reply])

    score = adapter.score_with_judge_contracts('edited', 'edited', '', _gedit_state())

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.judge_summary.failures == {'parse_error': 1}


@pytest.mark.parametrize('reply', [
    '{"total_points":1e309}',
    '{"total_points":-1e309}',
    '{"total_points":"Infinity"}',
    '{"total_points":"-Infinity"}',
])
def test_plawbench_non_finite_points_are_excluded(reply: str) -> None:
    adapter = get_benchmark('plawbench', TaskConfig(
        model='m', datasets=['plawbench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
    ))
    adapter.llm_judge = ScriptedJudge([reply])
    state = make_state('Response', '')
    state.metadata = {'judge_type': 'legal_qa', 'prompt': 'Question', 'rubrics': '', 'max_points': 12}

    score = adapter.score_with_judge_contracts('Response', 'Response', '', state)

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.judge_summary.failures == {'parse_error': 1}


def test_plawbench_finite_numeric_string_remains_supported() -> None:
    adapter = get_benchmark('plawbench', TaskConfig(
        model='m', datasets=['plawbench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
    ))
    adapter.llm_judge = ScriptedJudge(['{"total_points":"12"}'])
    state = make_state('Response', '')
    state.metadata = {'judge_type': 'legal_qa', 'prompt': 'Question', 'rubrics': '', 'max_points': 12}

    score = adapter.score_with_judge_contracts('Response', 'Response', '', state)

    assert score.status.is_usable
    assert score.value['acc'] == 1.0


@pytest.mark.parametrize('reply', [
    '{"idx_0":0.5}',
    '{"idx_0":0.0}',
    '{"idx_0":1.0}',
    '{"idx_0":true}',
    '{"idx_0":false}',
    '{"idx_0":2}',
])
def test_wide_search_non_integer_or_out_of_range_column_scores_are_rejected(reply: str) -> None:
    contract = OutputContract(schema_model=_column_score_model(1))

    assert contract.parse('{"idx_0":0}').ok
    assert contract.parse('{"idx_0":1}').ok
    assert not contract.parse(reply).ok


def test_strict_judge_scoring_versions_are_advanced() -> None:
    assert SHARED_SCORING_VERSION == 'v1.3'
    expected_versions = {
        'deepsearchqa': 'v1.1',
        'gedit': 'v1.1',
        'job_bench': 'v1.1',
        'plawbench': 'v1.1',
    }
    for name, expected in expected_versions.items():
        adapter = get_benchmark(name, TaskConfig(model='m', datasets=[name]))
        assert adapter.benchmark_meta.evaluation_version == expected


def test_browsecomp_grader_describes_json_fields_instead_of_line_output() -> None:
    config = TaskConfig(model='m', datasets=['browsecomp'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark('browsecomp', config)
    judge = ScriptedJudge([
        '{"extracted_final_answer": "Shakespeare", "reasoning": "matches", '
        '"correct": "yes", "confidence": 100}'
    ])
    adapter.llm_judge = judge
    state = make_state('Explanation: known\nExact Answer: Shakespeare\nConfidence: 100%', 'Shakespeare')
    state.metadata = {'question': 'Who wrote Hamlet?'}

    score = adapter.calculate_metrics(state).score

    assert score.status.is_usable
    prompt = '\n'.join(message.text for message in judge.calls[0])
    assert 'Populate the following JSON fields' in prompt
    assert '\nextracted_final_answer:' not in prompt
    assert '\nreasoning:' not in prompt
    assert '\ncorrect:' not in prompt
    assert '\nconfidence:' not in prompt
    assert 'single JSON object and no other text' in prompt
