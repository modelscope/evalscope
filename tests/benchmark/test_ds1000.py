from pathlib import Path
from typing import Any, Dict

import pytest

import evalscope  # noqa: F401  # imported for benchmark and metric registration side effects
from evalscope.api.benchmark import DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.ds1000.ds1000_adapter import SYSTEM_PROMPT
from evalscope.benchmarks.ds1000.utils import build_test_program, extract_code, run_locally
from evalscope.config import TaskConfig

# A DS-1000 style problem that only needs the standard library: the solution must set `result` to the
# sorted list, and test_string forbids a `for` loop (like DS-1000's vectorization checks).
CODE_CONTEXT = r'''
import copy
import tokenize, io


def generate_test_case(test_case_id):
    data = [3, 1, 2]
    return data, sorted(copy.deepcopy(data))


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
data = test_input
[insert]
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)


def test_string(solution: str):
    tokens = []
    for token in tokenize.tokenize(io.BytesIO(solution.encode("utf-8")).readline):
        tokens.append(token.string)
    assert "for" not in tokens
'''

RECORD: Dict[str, Any] = {
    'prompt': 'Sort `data`.\nA:\n<code>\ndata = [3, 1, 2]\n</code>\nresult = ... # put solution in this variable\n'
    'BEGIN SOLUTION\n<code>\n',
    'reference_code': 'result = sorted(data)\n',
    'metadata': {
        'problem_id': 7,
        'library_problem_id': 7,
        'library': 'Numpy',
        'test_case_cnt': 1,
        'perturbation_type': 'Origin',
        'perturbation_origin_id': 7,
    },
    'code_context': CODE_CONTEXT,
}


def _adapter() -> DefaultDataAdapter:
    return get_benchmark('ds1000', TaskConfig(datasets=['ds1000']))


def _score(adapter: DefaultDataAdapter, sample: Sample, completion: str):
    state = TaskState(
        model='mock',
        sample=sample,
        messages=[],
        output=ModelOutput.from_content(model='mock', content=completion),
        completed=True,
    )
    return adapter.calculate_metrics(state).score


def test_registered_meta() -> None:
    adapter = _adapter()

    assert adapter.subset_list == ['Pandas', 'Numpy', 'Matplotlib', 'Sklearn', 'Scipy', 'Pytorch', 'Tensorflow']
    assert adapter.eval_split == 'test'
    assert adapter.metric_list == ['accuracy']
    assert adapter.review_timeout == 120
    assert adapter.system_prompt == SYSTEM_PROMPT


def test_record_to_sample() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(dict(RECORD))
    messages = sample.input

    assert sample.subset_key == 'Numpy'
    assert sample.target == RECORD['reference_code']
    assert sample.metadata['perturbation_type'] == 'Origin'
    assert isinstance(messages[-1], ChatMessageUser)
    assert messages[-1].content.endswith('BEGIN SOLUTION\n<code>\n')


def test_system_prompt_is_prepended() -> None:
    adapter = _adapter()
    adapter.test_dataset = {'Numpy': [adapter.record_to_sample(dict(RECORD))]}
    adapter._post_process_samples()
    messages = adapter.test_dataset['Numpy'][0].input

    assert isinstance(messages[0], ChatMessageSystem)
    assert messages[0].content == SYSTEM_PROMPT


@pytest.mark.parametrize(
    'completion',
    [
        'result = sorted(data)\n</code>',
        'Here you go:\n<code>\nresult = sorted(data)\n</code>\nThis sorts the list.',
        'Use sorted:\n```python\nresult = sorted(data)\n```\nThat is all.',
        '```python\nresult = sorted(data)\n',  # unterminated fence
        '<think>maybe a loop</think>result = sorted(data)\nEND SOLUTION',
        'result = sorted(data)\n# SOLUTION END\nprint(1)',
    ],
)
def test_extract_code(completion: str) -> None:
    assert extract_code(completion).strip() == 'result = sorted(data)'


def test_build_test_program_adds_test_string_only_when_defined() -> None:
    program = build_test_program(CODE_CONTEXT, 'result = sorted(data)')
    assert program.endswith("code = 'result = sorted(data)'\ntest_execution(code)\ntest_string(code)\n")

    plain = build_test_program('def test_execution(code):\n    pass\n', 'x = 1')
    assert 'test_string(' not in plain.split('def test_execution')[1]


def test_run_locally_pass_fail_timeout() -> None:
    assert run_locally(build_test_program(CODE_CONTEXT, 'result = sorted(data)'), timeout=60)['passed']
    wrong = run_locally(build_test_program(CODE_CONTEXT, 'result = data'), timeout=60)
    assert not wrong['passed'] and 'AssertionError' in wrong['stderr']
    loop = run_locally(build_test_program(CODE_CONTEXT, 'result = [x for x in sorted(data)]'), timeout=60)
    assert not loop['passed']  # rejected by test_string
    assert run_locally('while True:\n    pass\n', timeout=1)['status'] == 'timeout'


def test_run_locally_uses_a_fresh_working_directory() -> None:
    program = 'import os\nassert not os.path.exists("output.png")\nopen("output.png", "w").write("x")\n'
    assert run_locally(program, timeout=60)['passed']
    assert run_locally(program, timeout=60)['passed']


def test_scoring_and_aggregation() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(dict(RECORD))
    completions = ['<code>\nresult = sorted(data)\n</code>', 'result = data', '', 'result = sorted(data)']
    scores = [SampleScore(score=_score(adapter, sample, c), sample_id=i) for i, c in enumerate(completions)]

    assert [s.score.value['accuracy'] for s in scores] == [True, False, False, True]
    aggregates = {agg.metric_name: agg for agg in adapter.aggregate_scores(scores)}
    assert aggregates['accuracy'].score == pytest.approx(0.5)


def test_bundled_dockerfile_covers_the_official_environment() -> None:
    # Every third-party import in the dataset's code_context fields must be installable from the image.
    spec = _adapter().get_sandbox_image_spec()
    dockerfile = Path(spec.dockerfile)
    assert dockerfile.is_file()
    content = dockerfile.read_text(encoding='utf-8')
    for package in (
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'xgboost',
        'gensim',
        'torch',
        'tensorflow-cpu',
    ):
        assert package in content
