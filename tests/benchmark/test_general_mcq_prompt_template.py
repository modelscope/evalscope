"""general_mcq keeps a prompt_template passed in dataset_args; only the default template is swapped
for the multiple-answer / chain-of-thought variants."""
import pytest

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.utils.multi_choices import MultipleChoiceTemplate

CUSTOM = 'Answer the question.\n{question}\n{choices}\nANSWER: [LETTER] from {letters}'


def _adapter(**dataset_args):
    return get_benchmark(
        'general_mcq',
        TaskConfig(model='mock', datasets=['general_mcq'], dataset_args={'general_mcq': dataset_args}),
    )


@pytest.mark.parametrize('extra_params', [{}, {'multiple_correct': True}, {'use_cot': True}])
def test_custom_prompt_template_is_kept(extra_params):
    assert _adapter(prompt_template=CUSTOM, extra_params=extra_params).prompt_template == CUSTOM


def test_default_template_still_follows_extra_params():
    assert _adapter().prompt_template == MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE
    assert (
        _adapter(extra_params={'multiple_correct': True}).prompt_template ==
        MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE
    )
