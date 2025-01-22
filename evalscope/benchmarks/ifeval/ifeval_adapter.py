from collections import defaultdict
from typing import Any, Dict, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.benchmarks.ifeval.utils import agg_inst_level_acc, process_results
from evalscope.constants import EvalType
from evalscope.metrics import Metric, mean
from evalscope.models import ChatGenerationModelAdapter


@Benchmark.register(
    name='ifeval',
    dataset_id='opencompass/ifeval',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['default'],
    metric_list=[
        Metric(name='prompt_level_strict_acc', object=mean),
        Metric(name='inst_level_strict_acc', object=agg_inst_level_acc),
        Metric(name='prompt_level_loose_acc', object=mean),
        Metric(name='inst_level_loose_acc', object=agg_inst_level_acc),
    ],
    few_shot_num=0,
    train_split=None,
    eval_split='train',
    prompt_template='',
)
class IFEvalAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> Any:
        return {'data': [input_d['prompt']], 'system_prompt': self.prompt_template}

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        return result

    def match(self, gold: Any, pred: Any) -> Dict:
        return process_results(gold, [pred])

    def compute_metric(self, review_res_list: List[dict]) -> Any:
        # aggregate review results
        res_dict = defaultdict(list)
        for res in review_res_list:
            for k, v in res.items():
                res_dict[k].append(v)

        metrics = []
        for metric in self.metric_list:
            metric_name = metric.name
            pred_value = res_dict[metric_name]
            metrics.append({'metric_name': metric_name, 'score': metric.object(pred_value), 'num': len(pred_value)})
        return metrics
