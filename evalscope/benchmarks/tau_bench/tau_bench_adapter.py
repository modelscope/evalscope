import importlib
from collections import defaultdict
from typing import Dict, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import Metric, mean, metric_registry
from evalscope.utils import get_logger

logger = get_logger()


@Benchmark.register(
    name='tau_bench',
    pretty_name='τ-bench',
    tags=['Reasoning', 'Agent', 'Function Calling'],
    description='A benchmark emulating dynamic conversations between a user (simulated by language models) '
    'and a language agent provided with domain-specific API tools and policy guidelines. '
    'Please install it with `pip install git+https://github.com/sierra-research/tau-bench` '
    'before evaluating and set a user model. [Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau_bench.html)',  # noqa: E501
    dataset_id='https://github.com/sierra-research/tau-bench',
    model_adapter='tau_bench_server',
    subset_list=['airline', 'retail'],
    metric_list=['Pass^1'],
    eval_split='test',
    extra_params={
        'user_model': 'qwen-plus',
        'api_key': 'EMPTY',
        'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'generation_config': {
            'temperature': 0.7,
            'max_new_tokens': 1024
        }
    })
class TauBenchAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        spec = importlib.util.find_spec('tau_bench')
        if spec is None:
            raise ImportError(
                '`tau_bench` not found, please install it with `pip install git+https://github.com/sierra-research/tau-bench` before evaluating.'  # noqa: E501
            )

        metric_registry.register(Metric(name='Pass^1', object=mean))

        # setup user model args
        extra_params = kwargs.get('extra_params', {})
        self.user_model = extra_params.get('user_model', 'qwen-plus')
        self.api_key = extra_params.get('api_key', 'EMPTY')
        self.api_base = extra_params.get('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.generation_config = extra_params.get('generation_config', {'temperature': 0.7, 'max_new_tokens': 1024})

        self._patch_env_completion()

    def _patch_env_completion(self) -> str:
        from tau_bench.envs.user import LLMUserSimulationEnv

        def new_generate_next_message(self, messages):
            from evalscope.models import ServerModelAdapter

            user_server = ServerModelAdapter(
                api_url=adapter_instance.api_base,
                model_id=adapter_instance.user_model,
                api_key=adapter_instance.api_key)
            request_json = user_server.make_request(
                input_item={'messages': messages}, infer_cfg=adapter_instance.generation_config)
            res = user_server.send_request(request_json)

            message = res['choices'][0]['message']
            self.messages.append(message)
            self.total_cost = 0
            return message['content']

        # get the current instance of TauBenchAdapter
        adapter_instance = self
        LLMUserSimulationEnv.generate_next_message = new_generate_next_message

    def load(self, **kwargs):
        from tau_bench.envs import get_env

        data_dict = defaultdict(dict)
        for env_name in self.subset_list:
            logger.info(f'Loading TauBench environment: {env_name}')
            env = get_env(
                env_name=env_name,
                user_strategy='llm',
                user_model='dummy',  # Use dummy model to prevent errors
                user_provider='openai',  # Use dummy provider to prevent errors
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
        import json
        res = json.loads(pred)
        return res.get('reward', 0.0)
