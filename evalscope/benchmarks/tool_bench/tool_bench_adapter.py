from typing import Dict, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import Metric, mean, metric_registry


@Benchmark.register(
    name='tool_bench',
    pretty_name='ToolBench-Static',
    dataset_id='AI-ModelScope/ToolBench-Static',
    subset_list=['in_domain', 'out_of_domain'],
    metric_list=['action_em', 'plan_em', 'f1', 'hallu_rate', 'rouge'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
)
class ToolBenchAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        metric_registry.register(Metric(name='rouge', object=mean))
        metric_registry.register(Metric(name='plan_em', object=mean))
        metric_registry.register(Metric(name='action_em', object=mean))
        metric_registry.register(Metric(name='f1', object=mean))
        metric_registry.register(Metric(name='hallu_rate', object=mean))

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from input data.
        """
        messages = input_d['messages']
        # use prepared messages
        return self.gen_prompt_data(prompt='', messages=messages)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return input_d

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        return result

    def match(self, gold: dict, pred: str) -> Dict:
        """
        Match the gold answer and the predicted answer.
        """
        from .utils import calculate_metrics

        data = {
            'target': gold['target'],
            'predictions': pred,
            'tools': gold['tools'],
        }
        metrics = calculate_metrics(data)
        return metrics

    def compute_metric(self, review_res_list: List[dict], **kwargs) -> Dict:
        # aggregate review results
        res_dict = super().compute_dict_metric(review_res_list, **kwargs)

        return super().compute_metric(res_dict, **kwargs)
