"""C-Eval, MaritimeBench, TruthfulQA and MMMLU format prompts with a prompt_template passed in dataset_args.

C-Eval and MaritimeBench declared their default template in BenchmarkMeta, but formatted prompts from the module
constant. TruthfulQA replaced the template in __init__, and MMMLU replaced it with the subset's language prompt, so a
prompt_template set through dataset_args was accepted and then ignored.
"""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.ceval.ceval_adapter import USER_PROMPT_TEMPLATE
from evalscope.benchmarks.maritime_bench.maritime_bench_adapter import MARITIME_PROMPT_TEMPLATE
from evalscope.benchmarks.mmmlu.prompt import LANGUAGE_PROMPT_MAP
from evalscope.config import TaskConfig
from evalscope.utils.multi_choices import MultipleChoiceTemplate

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


MC_CUSTOM = 'Q: {question}\n{choices}\nReply with one of {letters}.'


@pytest.mark.parametrize('multiple_correct', [False, True])
def test_truthful_qa_dataset_args_prompt_template_is_used(multiple_correct):
    adapter = _adapter('truthful_qa', prompt_template=MC_CUSTOM, extra_params={'multiple_correct': multiple_correct})
    sample = Sample(input='1+1=?', choices=['1', '2'], target=['B'])
    assert adapter.format_prompt_template(sample) == 'Q: 1+1=?\nA) 1\nB) 2\nReply with one of A,B.'


@pytest.mark.parametrize(
    'multiple_correct, default',
    [(False, MultipleChoiceTemplate.SINGLE_ANSWER), (True, MultipleChoiceTemplate.MULTIPLE_ANSWER)],
)
def test_truthful_qa_default_prompt_is_unchanged(multiple_correct, default):
    adapter = _adapter('truthful_qa', extra_params={'multiple_correct': multiple_correct})
    assert adapter.prompt_template == default


@pytest.mark.parametrize('language', ['ZH_CN', 'FR_FR'])
def test_mmmlu_dataset_args_prompt_template_is_used(language):
    adapter = _adapter('mmmlu', prompt_template=MC_CUSTOM)
    sample = Sample(input='1+1=?', choices=['1', '2', '3', '4'], target='B', metadata={'language': language})
    assert adapter.format_prompt_template(sample) == 'Q: 1+1=?\nA) 1\nB) 2\nC) 3\nD) 4\nReply with one of A,B,C,D.'


@pytest.mark.parametrize('language', ['ZH_CN', 'FR_FR'])
def test_mmmlu_dataset_args_default_template_is_used(language):
    """A configured template equal to the declared default is still used as given."""
    adapter = _adapter('mmmlu', prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT)
    sample = Sample(input='1+1=?', choices=['1', '2', '3', '4'], target='B', metadata={'language': language})
    prompt = adapter.format_prompt_template(sample)
    assert prompt.startswith(MultipleChoiceTemplate.SINGLE_ANSWER_COT.split('{')[0])


@pytest.mark.parametrize('dataset_args', [{}, {'prompt_template': None}])
@pytest.mark.parametrize('language', ['ZH_CN', 'FR_FR'])
def test_mmmlu_default_prompt_uses_subset_language(language, dataset_args):
    adapter = _adapter('mmmlu', **dataset_args)
    sample = Sample(input='1+1=?', choices=['1', '2', '3', '4'], target='B', metadata={'language': language})
    prompt = adapter.format_prompt_template(sample)
    assert prompt.startswith(LANGUAGE_PROMPT_MAP[language].split('{')[0])
