from typing import Any

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

MARITIME_PROMPT_TEMPLATE = '请回答单选题。要求只输出选项，不输出解释，将选项放在[]里，直接输出答案。示例：\n\n题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。\n选项：\nA. 电磁力\nB. 压拉应力\nC. 弯曲应力\nD. 扭应力\n答：[A]\n 当前题目\n {question}\n选项：\n{choices}'  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='maritime_bench',
        pretty_name='MaritimeBench',
        tags=[Tags.CHINESE, Tags.MULTIPLE_CHOICE, Tags.KNOWLEDGE],
        description=
        'MaritimeBench is a benchmark for evaluating AI models on maritime-related multiple-choice questions. It consists of questions related to maritime knowledge, where the model must select the correct answer from given options.',  # noqa: E501
        dataset_id='HiDolphin/MaritimeBench',
        metric_list=['acc'],
        few_shot_num=0,
        eval_split='test',
        prompt_template=MARITIME_PROMPT_TEMPLATE,
    )
)
class MaritimeBenchAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record) -> Sample:
        # Extract available choices from the record
        choices = []
        choice_letters = ['A', 'B', 'C', 'D']
        for letter in choice_letters:
            if letter in record and record[letter]:
                choices.append(record[letter])

        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],
        )

    def format_prompt_template(self, sample):
        choices = '\n'.join([f'{chr(65 + i)}. {choice}' for i, choice in enumerate(sample.choices)])
        return MARITIME_PROMPT_TEMPLATE.format(question=sample.input, choices=choices)

    def extract_answer(self, prediction, task_state):
        # use regex to extract the answer from the prediction
        import re
        match = re.search(r'\[([A-D])\]', prediction)
        if match:
            return match.group(1)
        return ''
