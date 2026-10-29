# Copyright (c) Alibaba, Inc. and its affiliates.
"""SciCode-Verified follows the SciCode-Verified harness: official prompts, reference code for the skipped steps,
official code extraction, and OR grading over two pinned environments. Offline: no dataset, model or sandbox."""
import base64
import re
import subprocess
import sys

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.scicode_verified.scicode_verified_adapter import OR_WRAPPER, SciCodeVerifiedAdapter
from evalscope.config import TaskConfig
from evalscope.constants import HubType


def make_adapter(**extra_params) -> SciCodeVerifiedAdapter:
    cfg = TaskConfig(
        model='mock',
        eval_type='openai_api',
        api_url='http://127.0.0.1:1/v1',
        datasets=['scicode_verified'],
        dataset_args={'scicode_verified': {'extra_params': extra_params}},
    )
    return get_benchmark('scicode_verified', config=cfg)


def make_sample(problem_id: str = '62') -> Sample:
    steps = [{
        'step_number': f'{problem_id}.{k}',
        'step_description_prompt': f'Describe step {k}',
        'step_background': f'Background of step {k}',
        'function_header': f'def f{k}(x):',
        'return_line': '    return y',
        'test_cases': [f'from scicode.compare.cmp import cmp_tuple_or_list\nassert f{k}(1) == target'],
    } for k in (1, 2, 3)]
    record = {'problem_id': problem_id, 'required_dependencies': 'import numpy as np', 'sub_steps': steps}
    return Sample(input=[ChatMessageUser(content=problem_id)], metadata=record)


class FakeModel:
    name = 'mock'

    def __init__(self):
        self.inputs = []

    def generate(self, input, tools=None):
        self.inputs.append(input)
        k = len(self.inputs)
        return ModelOutput.from_content('mock', f'Answer:\n```python\nimport os\ndef f_model_{k}(x):\n    return x\n```\n')


def decoded_script(wrapper: str) -> str:
    return base64.b64decode(re.search(r"b64decode\('([^']+)'\)", wrapper).group(1)).decode()


def test_extraction_follows_the_official_rule() -> None:
    extract = SciCodeVerifiedAdapter._extract
    # first python block, import lines removed, surrounding whitespace kept (it is pasted into the next prompt);
    # as the official regex `^\s*(import ...)`, the removal also takes the newline before the import line
    assert extract('x\n```python\nimport numpy as np\ndef f():\n    return 1\n```\n```python\ndef g(): pass\n```') == (
        '\ndef f():\n    return 1\n')
    assert extract('```\ndef f(): pass\n```') == '\ndef f(): pass\n'
    assert extract('def f(): pass') == 'def f(): pass'
    assert extract('') == ''


@pytest.mark.parametrize('background', [False, True])
def test_official_prompt_lists_previous_steps_and_code(background: bool) -> None:
    adapter = make_adapter(provide_background=background)
    sample = make_sample()

    prompt = adapter._official_prompt(sample, 2, ['CODE_1', 'CODE_2'])

    expected_steps = '\n\n'.join([
        'Describe step 1' + ('\nBackground of step 1' if background else ''), 'CODE_1', '------',
        'Describe step 2' + ('\nBackground of step 2' if background else ''), 'CODE_2'])
    assert expected_steps in prompt
    next_step = 'Describe step 3' + ('\nBackground of step 3' if background else '') + '\n\ndef f3(x):\n\n    return y'
    assert next_step in prompt
    assert ('PROBLEM STEPS AND FUNCTION CODE' in prompt) == background  # multistep vs background-comment template
    assert 'import numpy as np' in prompt


@pytest.mark.parametrize('prompt_mode', ['official', 'multi_turn'])
def test_skipped_step_gets_reference_code_and_is_not_scored(prompt_mode: str, monkeypatch) -> None:
    adapter = make_adapter(prompt_mode=prompt_mode)
    sample, model = make_sample('62'), FakeModel()

    state = adapter.run_inference(model, sample, output_dir='/tmp')

    assert len(model.inputs) == 2  # 62.1 is not generated
    if prompt_mode == 'official':
        assert all(len(messages) == 1 for messages in model.inputs)  # single-turn requests
    reference = adapter._gold_code('62.1').strip().splitlines()[0]
    assert any(reference in m.text for m in model.inputs[0])  # the next request sees the reference code
    code = decoded_script(adapter._compose_code(sample.metadata['sub_steps'][2], state))
    assert reference in code and 'f_model_1' in code and 'f_model_2' in code
    assert 'import os' not in code and 'scicode.compare' not in code
    assert "process_hdf5_to_tuple('62.3', 1)" in code

    monkeypatch.setattr(SciCodeVerifiedAdapter, '_verify_subproblem',
                        lambda self, subproblem, state: Score(value={subproblem['step_number']: 1.0}))
    score = adapter.match_score('', '', '', state)
    assert set(score.value) == {'main_problem', '62.2', '62.3'} and score.value['main_problem'] == 1.0


def test_grading_environments() -> None:
    assert [label for label, _ in make_adapter().grading_envs] == ['2024', '2025']
    assert [label for label, _ in make_adapter(grading_envs='2025').grading_envs] == ['2025']
    with pytest.raises(ValueError):
        make_adapter(grading_envs='2026')


def run_wrapper(script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    wrapper = OR_WRAPPER.format(
        b64=base64.b64encode(script.encode()).decode(),
        envs=[('2024', sys.executable), ('2025', sys.executable)],
        timeout=timeout,
    )
    return subprocess.run([sys.executable, '-c', wrapper], capture_output=True, text=True, timeout=60)


def test_or_wrapper_passes_on_the_first_passing_environment() -> None:
    marker = '/tmp/scicode_verified_or_test_marker'
    # fails in the first environment run, passes in the second
    script = (f'import os\nif not os.path.exists({marker!r}):\n    open({marker!r}, "w").close()\n'
              f'    raise SystemExit(1)\nos.remove({marker!r})')
    result = run_wrapper(script)
    assert result.returncode == 0
    assert re.findall(r'SCV_ENV (\S+) (PASS|FAIL)', result.stdout) == [('2024', 'FAIL'), ('2025', 'PASS')]

    result = run_wrapper('pass')
    assert result.returncode == 0 and re.findall(r'SCV_ENV (\S+) (PASS|FAIL)', result.stdout) == [('2024', 'PASS')]


def test_or_wrapper_fails_with_the_last_output_and_reports_timeouts() -> None:
    result = run_wrapper('raise ValueError("wrong shape")')
    assert result.returncode == 1 and 'ValueError: wrong shape' in result.stdout

    result = run_wrapper('import time\ntime.sleep(5)', timeout=1)
    assert result.returncode == 1 and 'Timeout: step exceeded 1 s' in result.stdout
    assert len(re.findall(r'SCV_ENV \S+ FAIL [0-9.]+s', result.stdout)) == 2


@pytest.mark.parametrize(
    'status, output, passed, per_env, timed_out',
    [
        ('success', 'SCV_ENV 2024 PASS 3.1s\n', True, {'2024': True}, False),
        ('error', 'SCV_ENV 2024 FAIL 1800.0s\nSCV_ENV 2025 FAIL 2.5s\nValueError\n', False,
         {'2024': False, '2025': False}, True),
        ('success', 'SCV_ENV 2024 FAIL 1800.0s\nSCV_ENV 2025 PASS 900.2s\n', True, {'2024': False, '2025': True}, False),
        ('error', '', False, {}, False),
    ],
)
def test_grading_metadata(status, output, passed, per_env, timed_out, monkeypatch) -> None:
    adapter = make_adapter()
    sample = make_sample('5')
    state = adapter.run_inference(FakeModel(), sample, output_dir='/tmp')
    monkeypatch.setattr(SciCodeVerifiedAdapter, 'execute_code_in_sandbox',
                        lambda self, code, timeout, language: {'status': status, 'output': output})

    score = adapter._verify_subproblem(sample.metadata['sub_steps'][0], state)

    assert (score.value['5.1'] == 1.0) is passed
    assert score.metadata['per_env'] == per_env and score.metadata['timed_out'] is timed_out
    assert score.metadata['grading_seconds_total'] == round(sum(score.metadata['grading_seconds'].values()), 1)


def test_problem_filter_and_hub() -> None:
    adapter = make_adapter(problem_ids=['12', 62])
    assert adapter.sample_filter(make_sample('62')) and not adapter.sample_filter(make_sample('5'))
    assert make_adapter().sample_filter(make_sample('5'))
    assert adapter.dataset_hub == HubType.HUGGINGFACE  # SciCode-Verified is only published on the HF hub
