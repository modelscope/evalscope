from evalscope.benchmarks.mm_safetybench_plus_plus.mm_safetybench_plus_plus_adapter import (
    QUALITY_CONTRACT,
    MMSafetyBenchPlusPlusAdapter,
)
from evalscope.benchmarks.mm_safetybench_plus_plus.utils import build_safe_judge_prompt, build_unsafe_judge_prompt


def test_unsafe_prompt_contains_rubric_and_response():
    prompt = build_unsafe_judge_prompt('Sure, here are the steps...')
    assert 'RQ-5 Rubric' in prompt
    assert 'expert AI safety evaluator' in prompt
    assert 'Sure, here are the steps...' in prompt
    # No unrendered placeholder left behind.
    assert '{model_response}' not in prompt


def test_safe_prompt_contains_rubric_and_response():
    prompt = build_safe_judge_prompt('To reset your password, ...')
    assert 'RQ-5 Rubric' in prompt
    assert 'expert AI helpfulness evaluator' in prompt
    assert 'To reset your password, ...' in prompt
    assert '{model_response}' not in prompt


def test_prompts_diverge_by_branch():
    # The two rubrics must be distinct so the judge prompt cannot be applied to the wrong branch.
    assert build_unsafe_judge_prompt('x') != build_safe_judge_prompt('x')


def test_contract_parses_integer_reply():
    result = QUALITY_CONTRACT.parse('{"score": 4}')
    assert result.ok
    assert result.value.score == 4


def test_contract_rejects_out_of_range_and_prose():
    assert not QUALITY_CONTRACT.parse('{"score": 6}').ok
    assert not QUALITY_CONTRACT.parse('{"score": -1}').ok
    assert not QUALITY_CONTRACT.parse('The response is good, score 5').ok
    assert not QUALITY_CONTRACT.parse('').ok


def test_record_to_sample_skips_unknown_label():
    adapter = MMSafetyBenchPlusPlusAdapter.__new__(MMSafetyBenchPlusPlusAdapter)
    assert adapter.record_to_sample({'label': 'weird', 'image': None}) is None
    assert adapter.record_to_sample({'label': 'unsafe', 'image': None}) is None
