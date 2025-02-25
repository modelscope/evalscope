from collections import defaultdict
from typing import Any, Dict, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.benchmarks.ifeval.utils import process_results
from evalscope.constants import EvalType
from evalscope.metrics import Metric, mean, metric_registry
from evalscope.models import ChatGenerationModelAdapter


@Benchmark.register(
    name='ifeval',
    dataset_id='opencompass/ifeval',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['default'],
    metric_list=[
        'prompt_level_strict_acc',
        'inst_level_strict_acc',
        'prompt_level_loose_acc',
        'inst_level_loose_acc',
    ],
    few_shot_num=0,
    train_split=None,
    eval_split='train',
    prompt_template='',
)
class IFEvalAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # register metrics
        metric_registry.register(Metric(name='prompt_level_strict_acc', object=mean))
        metric_registry.register(Metric(name='inst_level_strict_acc', object=mean))
        metric_registry.register(Metric(name='prompt_level_loose_acc', object=mean))
        metric_registry.register(Metric(name='inst_level_loose_acc', object=mean))

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> Any:
        return {'data': [input_d['prompt']], 'system_prompt': self.system_prompt}

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        return result

    def match(self, gold: Any, pred: Any) -> Dict:
        return process_results(gold, [pred])

    def compute_metric(self, review_res_list: List[dict], **kwargs) -> Any:
        # aggregate review results
        res_dict = defaultdict(list)
        for res in review_res_list:
            for k, v in res.items():
                res_dict[k].append(v)

        return super().compute_metric(res_dict)
