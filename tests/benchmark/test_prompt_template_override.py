"""C-Eval and MaritimeBench format prompts with a prompt_template passed in dataset_args.

Both adapters declare their default template in BenchmarkMeta, but formatted prompts from the module constant,
so a prompt_template set through dataset_args was accepted and then ignored.
"""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.ceval.ceval_adapter import USER_PROMPT_TEMPLATE
from evalscope.benchmarks.maritime_bench.maritime_bench_adapter import MARITIME_PROMPT_TEMPLATE
from evalscope.config import TaskConfig

CEVAL_CUSTOM = 'Subject: {subject}\nQ: {question}\n{choices}\nReply with 答案：X'
MARITIME_CUSTOM = 'Q: {question}\n{choices}\nPut the option in [].'


def _adapter(name: str, **dataset_args):
    return get_benchmark(name, TaskConfig(model='mock', datasets=[name], dataset_args={name: dataset_args}))


def _ceval_sample() -> Sample:
    return Sample(input='1+1=?', choices=['1', '2', '3', '4'], target='B', metadata={'subject': 'computer_network'})


def _maritime_sample() -> Sample:
    return Sample(input='1+1=?', choices=['1', '2', '3', '4'], target='B')


@pytest.mark.parametrize(
    'name, template, sample, expected',
    [
        ('ceval', CEVAL_CUSTOM, _ceval_sample(), 'Subject: 计算机网络\nQ: 1+1=?\nA. 1\nB. 2\nC. 3\nD. 4\nReply with 答案：X'),
        ('maritime_bench', MARITIME_CUSTOM, _maritime_sample(), 'Q: 1+1=?\nA. 1\nB. 2\nC. 3\nD. 4\nPut the option in [].'),
    ],
)
def test_dataset_args_prompt_template_is_used(name, template, sample, expected):
    adapter = _adapter(name, prompt_template=template)
    assert adapter.prompt_template == template
    assert adapter.format_prompt_template(sample) == expected


@pytest.mark.parametrize(
    'name, default, sample',
    [('ceval', USER_PROMPT_TEMPLATE, _ceval_sample()), ('maritime_bench', MARITIME_PROMPT_TEMPLATE, _maritime_sample())],
)
def test_default_prompt_is_unchanged(name, default, sample):
    adapter = _adapter(name)
    assert adapter.prompt_template == default
    prompt = adapter.format_prompt_template(sample)
    assert prompt.startswith(default.split('{')[0])
    assert '1+1=?' in prompt and 'D. 4' in prompt
