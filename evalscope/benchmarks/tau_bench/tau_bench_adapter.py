import importlib
from collections import defaultdict
from typing import Dict, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import Metric, mean, metric_registry


@Benchmark.register(
    name='tau_bench',
    pretty_name='τ-bench',
    model_adapter='tau_bench',
    tags=['Reasoning', 'Agent', 'Function Calling'],
    description='A benchmark emulating dynamic conversations between a user (simulated by language models) '
    'and a language agent provided with domain-specific API tools and policy guidelines. '
    'Please install it with `pip install git+https://github.com/sierra-research/tau-bench` before evaluating. '
    '[Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)',  # noqa: E501
    dataset_id='https://github.com/sierra-research/tau-bench',
    subset_list=['airline', 'retail'],
    metric_list=['Pass^1'],
    eval_split='test',
)
class TauBenchAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        spec = importlib.util.find_spec('tau_bench')
        if spec is None:
            raise ImportError(
                '`tau_bench` not found, please install it with `pip install git+https://github.com/sierra-research/tau-bench` before evaluating.'  # noqa: E501
            )

        metric_registry.register(Metric(name='Pass^1', object=mean))

        self.llm_as_a_judge = True

    def load(self, **kwargs):
        from tau_bench.envs import get_env

        data_dict = defaultdict(dict)
        for env_name in self.subset_list:
            env = get_env(
                env_name=env_name,
                user_strategy='llm',
                user_model='gpt-4o',
                task_split=self.eval_split,
            )
            tasks = []
            for i in range(len(env.tasks)):
                tasks.append({
                    'task_index': i,
                    'env_name': env_name,
                })
            data_dict[env_name][self.eval_split] = tasks

        return data_dict

    def gen_prompt(self, input_d, subset_name, few_shot_list, **kwargs):
        return self.gen_prompt_data(extra_data=input_d)

    def get_gold_answer(self, input_d):
        return ''

    def match(self, gold, pred):
        return super().match(gold, pred)
