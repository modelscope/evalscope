"""Few-shot demonstrations must use the answer format the Chinese prompt asks for and the parser reads.

With a Chinese answer template, MultiChoiceAdapter asks for '答案：...' and parses replies with
parse_answers_zh, so its default demonstrations must end with '答案：...' as well, under a Chinese
header. Otherwise a model that copies the demonstrated 'ANSWER: AC' is read as the last capital only.
"""
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


def test_multiple_correct_demonstrations_parse_back_to_their_targets():
    adapter = _adapter(subset_list=['example_multi'], extra_params={'multiple_correct': True})
    assert adapter.prompt_template == MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE
    dataset = adapter.load_dataset()
    prompt = dataset['example_multi'][0].input[-1].content

    demonstrations = adapter.fewshot_dataset['example_multi']
    assert len(demonstrations) == 2
    for sample in demonstrations:
        demonstration = adapter.sample_to_fewshot(sample)
        assert demonstration in prompt
        answer_line = _answer_line(demonstration)
        assert _extract(adapter, sample, answer_line) == ''.join(sorted(sample.target))
        assert answer_line.startswith('答案：')

    assert prompt.startswith(MultipleChoiceTemplate.CHINESE_FEW_SHOT_TEMPLATE.split('\n')[0])
    assert 'ANSWER:' not in prompt


def test_single_answer_demonstrations_use_the_chinese_marker():
    adapter = _adapter(subset_list=['example'])
    dataset = adapter.load_dataset()
    prompt = dataset['example'][0].input[-1].content
    for sample in adapter.fewshot_dataset['example']:
        answer_line = _answer_line(adapter.sample_to_fewshot(sample))
        assert answer_line == f'答案：{sample.target}'
        assert _extract(adapter, sample, answer_line) == sample.target
    assert 'ANSWER:' not in prompt


def test_english_template_demonstrations_are_unchanged():
    template = MultipleChoiceTemplate.SINGLE_ANSWER
    adapter = _adapter(subset_list=['example'], prompt_template=template)
    dataset = adapter.load_dataset()
    prompt = dataset['example'][0].input[-1].content
    sample = adapter.fewshot_dataset['example'][0]
    demonstration = adapter.sample_to_fewshot(sample)
    assert _answer_line(demonstration) == f'ANSWER: {sample.target}'
    assert prompt.startswith('Here are some examples of how to answer similar questions:')
