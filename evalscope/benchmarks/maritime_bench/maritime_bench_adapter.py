from typing import Any

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import exact_match
from evalscope.utils.utils import ResponseParser

SUBSET_LIST = ['default']


@Benchmark.register(
    name='maritime_bench',
    pretty_name='MaritimeBench',
    dataset_id='HiDolphin/MaritimeBench',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.MULTIPLE_CHOICE, OutputType.GENERATION],
    subset_list=SUBSET_LIST,
    metric_list=['AverageAccuracy'],
    eval_split='test',
    prompt_template=
    '题目来自于{subset_name}请回答单选题。要求只输出选项，不输出解释，将选项放在<>里，直接输出答案。示例：\n\n题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。\n选项：\nA. 电磁力\nB. 压拉应力\nC. 弯曲应力\nD. 扭应力\n答：<A> 当前题目\n {query}',  # noqa: E501
)
class MaritimeBenchAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D']

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> Any:

        prefix = ''
        query = prefix + input_d['question'] + '\n'
        available_choices = []
        for option in self.choices:
            if option in input_d and input_d[option]:
                query += option + ':' + input_d[option] + '\n'
                available_choices.append(option)

        full_prompt = self.prompt_template.format(subset_name=subset_name, query=query)
        return self.gen_prompt_data(full_prompt, choices=available_choices)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).

        Args:
            input_d: input raw data. Depending on the dataset.

        Returns:
            The parsed input. e.g. gold answer ... Depending on the dataset.
        """
        return input_d['answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the raw model prediction (pred).

        Args:
            pred: model prediction. Depending on the model.

        Returns:
            The parsed prediction. e.g. model answer... Depending on the model.
        """

        return ResponseParser.parse_bracketed_answer(result, options=self.choices)

    def match(self, gold: Any, pred: Any) -> Any:
        """
        Match the gold answer with the predicted answer.

        Args:
            gold: The gold answer.
            pred: The predicted answer.

        Returns:
            The result of the match.
        """
        return exact_match(gold=gold, pred=pred)
