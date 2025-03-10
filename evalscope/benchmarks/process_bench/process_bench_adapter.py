import os
import re
from typing import Any, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import AnswerKeys, EvalType
from evalscope.metrics import Metric, mean, metric_registry, simple_f1_score

cur_path = os.path.dirname(os.path.abspath(__file__))


@Benchmark.register(
    name='process_bench',
    pretty_name='ProcessBench',
    dataset_id='Qwen/ProcessBench',
    subset_list=['gsm8k', 'math', 'olympiadbench', 'omnimath'],
    metric_list=['error_acc', 'correct_acc', 'simple_f1_score'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
)
class ProcessBenchAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prompt_template = open(os.path.join(cur_path, 'critique_template.txt'), encoding='utf-8').read()

        # register metrics
        metric_registry.register(Metric(name='error_acc', object=mean))
        metric_registry.register(Metric(name='correct_acc', object=mean))
        metric_registry.register(Metric(name='simple_f1_score', object=simple_f1_score))

    def load(self, **kwargs):
        # default load all levels
        kwargs['split_as_subset'] = True
        data_dict = super().load(**kwargs)
        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> Any:

        problem = input_d['problem']
        steps = input_d['steps']
        tagged_response = ''
        for sdx, step in enumerate(steps):
            tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
        tagged_response = tagged_response.strip()

        full_prompt = self.prompt_template.format(problem=problem, tagged_response=tagged_response)

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return int(input_d['label'])

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        pred = ProcessBenchAdapter.extract_answer(result)
        try:
            pred = int(pred)
        except Exception:
            pred = None
        return pred

    def match(self, gold: int, pred: int) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        return gold == pred

    def compute_metric(self, review_res_list: list, **kwargs) -> List[dict]:
        reviews_list = kwargs['reviews_list']
        error_data = []
        correct_data = []
        for res, raw in zip(review_res_list, reviews_list):
            if raw[AnswerKeys.RAW_INPUT]['label'] == -1:
                correct_data.append(res)
            else:
                error_data.append(res)
        data = {}
        if len(correct_data) != 0:
            data.update({'correct_acc': correct_data})
        if len(error_data) != 0:
            data.update({'error_acc': error_data})
        data.update({'simple_f1_score': (correct_data, error_data)})
        return super().compute_metric(data)

    @staticmethod
    def extract_answer(solution_text: str):
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(boxed_pattern, solution_text)
        if matches:
            return matches[-1].strip()
        return None
