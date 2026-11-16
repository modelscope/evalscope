"""Few-shot demonstrations must use the answer format the Chinese prompt asks for and the parser reads.

With a Chinese answer template, MultiChoiceAdapter asks for '答案：...' and parses replies with
parse_answers_zh, so its default demonstrations must end with '答案：...' as well, under a Chinese
header. Otherwise a model that copies the demonstrated 'ANSWER: AC' is read as the last capital only.
"""
import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.utils.multi_choices import MultipleChoiceTemplate

MCQ_PATH = 'custom_eval/text/mcq'


def _adapter(**dataset_args):
    return get_benchmark(
        'general_mcq',
        TaskConfig(
            model='mock',
            datasets=['general_mcq'],
            dataset_args={'general_mcq': {'local_path': MCQ_PATH, 'few_shot_num': 2, **dataset_args}},
            limit=1,
        ),
    )


def _answer_line(demonstration: str) -> str:
    return demonstration.strip().splitlines()[-1]


def _extract(adapter, sample, completion: str) -> str:
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content(model='mock', content=completion))
    return adapter.extract_answer(completion, state)


CUSTOM_OUTER = '自定义示例开始\n{fewshot}\n自定义示例结束\n\n问题：{question}\n{choices}\n请用“答案：X”作答。'

CHINESE_TEMPLATES = {
    (False, False): MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE,
    (False, True): MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT,
    (True, False): MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE,
    (True, True): MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT,
}


@pytest.mark.parametrize('custom_outer', [False, True], ids=['default-outer', 'custom-outer'])
@pytest.mark.parametrize('use_cot', [False, True], ids=['no-cot', 'cot'])
@pytest.mark.parametrize('multiple_correct', [False, True], ids=['single', 'multiple'])
def test_chinese_template_demonstrations(multiple_correct, use_cot, custom_outer):
    """The default demonstrations follow the Chinese answer template; a custom outer template is kept as given."""
    subset = 'example_multi' if multiple_correct else 'example'
    dataset_args = {
        'subset_list': [subset],
        'extra_params': {'multiple_correct': multiple_correct, 'use_cot': use_cot},
    }
    if custom_outer:
        dataset_args['few_shot_prompt_template'] = CUSTOM_OUTER
    adapter = _adapter(**dataset_args)
    assert adapter.prompt_template == CHINESE_TEMPLATES[(multiple_correct, use_cot)]
    prompt = adapter.load_dataset()[subset][0].input[-1].content

    demonstrations = adapter.fewshot_dataset[subset]
    assert len(demonstrations) == 2
    for sample in demonstrations:
        demonstration = adapter.sample_to_fewshot(sample)
        assert demonstration in prompt
        assert demonstration.startswith('问题：')
        answer_line = _answer_line(demonstration)
        assert answer_line.startswith('答案：')
        expected = ''.join(sorted(sample.target)) if multiple_correct else sample.target
        assert _extract(adapter, sample, answer_line) == expected

    if custom_outer:
        assert prompt.startswith('自定义示例开始\n')
        assert '自定义示例结束' in prompt
        assert MultipleChoiceTemplate.CHINESE_FEW_SHOT_TEMPLATE.split('\n')[0] not in prompt
    else:
        assert prompt.startswith(MultipleChoiceTemplate.CHINESE_FEW_SHOT_TEMPLATE.split('\n')[0])
    assert 'ANSWER:' not in prompt


def test_overridden_sample_to_fewshot_is_kept():
    """An adapter that overrides sample_to_fewshot keeps its own demonstrations under a Chinese template."""
    adapter = _adapter(subset_list=['example'])

    class CustomFewshotAdapter(type(adapter)):

        def sample_to_fewshot(self, sample):
            return f'Q: {sample.input}\nA: {sample.target}'

    custom = CustomFewshotAdapter(benchmark_meta=adapter._benchmark_meta, task_config=adapter._task_config)
    assert custom.prompt_template == MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE
    prompt = custom.load_dataset()['example'][0].input[-1].content
    for sample in custom.fewshot_dataset['example']:
        assert f'Q: {sample.input}\nA: {sample.target}' in prompt
        assert adapter.sample_to_fewshot(sample) not in prompt


def test_english_template_demonstrations_are_unchanged():
    template = MultipleChoiceTemplate.SINGLE_ANSWER
    adapter = _adapter(subset_list=['example'], prompt_template=template)
    dataset = adapter.load_dataset()
    prompt = dataset['example'][0].input[-1].content
    sample = adapter.fewshot_dataset['example'][0]
    demonstration = adapter.sample_to_fewshot(sample)
    assert _answer_line(demonstration) == f'ANSWER: {sample.target}'
    assert prompt.startswith('Here are some examples of how to answer similar questions:')
