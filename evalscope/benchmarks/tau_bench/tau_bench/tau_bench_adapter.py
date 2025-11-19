from collections import defaultdict
from typing import Dict, List

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.dataset.dataset import DatasetDict
from evalscope.api.dataset.loader import DictDataLoader
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils import get_logger
from evalscope.utils.function_utils import run_once
from evalscope.utils.import_utils import check_import

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='tau_bench',
        pretty_name='τ-bench',
        tags=[Tags.FUNCTION_CALLING, Tags.REASONING, Tags.AGENT],
        description='A benchmark emulating dynamic conversations between a user (simulated by language models) '
        'and a language agent provided with domain-specific API tools and policy guidelines. '
        'Please install it with `pip install git+https://github.com/sierra-research/tau-bench` '
        'before evaluating and set a user model. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/tau_bench.html)',  # noqa: E501
        dataset_id='https://github.com/sierra-research/tau-bench',
        subset_list=['airline', 'retail'],
        aggregation='mean_and_pass_hat_k',
        eval_split='test',
        extra_params={
            'user_model': {
                'type': 'str',
                'description': 'Model used to simulate the user in the environment.',
                'value': 'qwen-plus'
            },
            'api_key': {
                'type': 'str',
                'description': 'API key for the user model backend.',
                'value': 'EMPTY'
            },
            'api_base': {
                'type': 'str',
                'description': 'Base URL for the user model API requests.',
                'value': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            },
            'generation_config': {
                'type': 'dict',
                'description': 'Default generation config for user model simulation.',
                'value': {
                    'temperature': 0.0,
                }
            }
        }
    )
)
class TauBenchAdapter(AgentAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import(
            'tau_bench',
            package='git+https://github.com/sierra-research/tau-bench',
            raise_error=True,
            feature_name=self.pretty_name
        )

        # setup user model args
        self.user_model = self.extra_params.get('user_model', 'qwen-plus')
        self.api_key = self.extra_params.get('api_key', 'EMPTY')
        self.api_base = self.extra_params.get('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.generation_config = self.extra_params.get('generation_config', {'temperature': 0.0, 'max_tokens': 4096})

    @run_once
    def _patch_env_completion(self) -> str:
        from tau_bench.envs.user import LLMUserSimulationEnv

        def new_generate_next_message(self, messages):
            from evalscope.api.messages import dict_to_chat_message
            from evalscope.api.model import GenerateConfig, get_model
            from evalscope.constants import EvalType

            user_server = get_model(
                model=adapter_instance.user_model,
                eval_type=EvalType.SERVICE,
                base_url=adapter_instance.api_base,
                api_key=adapter_instance.api_key,
                config=GenerateConfig(**adapter_instance.generation_config)
            )

            res = user_server.generate(input=[dict_to_chat_message(msg) for msg in messages])

            message = {'role': 'assistant', 'content': res.completion}
            self.messages.append(message)
            self.total_cost = 0
            return res.completion

        # get the current instance of TauBenchAdapter
        adapter_instance = self
        LLMUserSimulationEnv.generate_next_message = new_generate_next_message

    def load(self):
        from tau_bench.envs import get_env

        self._patch_env_completion()

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
            # load dataset
            dataset = DictDataLoader(
                dict_list=tasks,
                sample_fields=self.record_to_sample,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
            ).load()

            data_dict[env_name] = dataset

        test_dataset = DatasetDict(data_dict)

        return test_dataset, None

    def record_to_sample(self, record: Dict) -> Sample:
        """Convert a data record to a Sample object."""
        return Sample(
            input=[ChatMessageUser(content='')],
            target='',  # Will use the record for evaluation
            subset_key=record['env_name'],
            metadata=record  # Store the full record for evaluation
        )

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        from .generation import predict
        return predict(model, sample)

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state) -> Score:

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        try:
            # Parse the prediction to get the reward
            task_result = task_state.metadata['task_result']
            reward = task_result.get('reward', 0.0)

            score.value = {
                'acc': float(reward),
            }
            score.explanation = f'Task completed with reward: {reward}'
            score.metadata = {
                'task_result': task_result,
                'env_name': task_state.metadata.get('env_name', 'unknown'),
                'task_index': task_state.metadata.get('task_index', -1)
            }
            score.main_score_name = 'acc'

        except Exception as e:
            score.value = {'acc': 0.0}
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata = {'error': str(e)}
            score.main_score_name = 'acc'

        return score
