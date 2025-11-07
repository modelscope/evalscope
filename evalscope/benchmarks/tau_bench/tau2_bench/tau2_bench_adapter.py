import os
from collections import defaultdict
from typing import Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
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
        name='tau2_bench',
        pretty_name='τ2-bench',
        tags=[Tags.FUNCTION_CALLING, Tags.REASONING, Tags.AGENT],
        description='A benchmark emulating dynamic conversations between a user (simulated by language models) '
        'and a language agent provided with domain-specific API tools and policy guidelines. '
        'Please install it with `pip install git+https://github.com/sierra-research/tau2-bench@v0.2.0` '
        'before evaluating and set a user model. [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/tau_bench.html)',  # noqa: E501
        dataset_id='evalscope/tau2-bench-data',
        subset_list=['airline', 'retail', 'telecom'],
        metric_list=['Pass^1'],
        eval_split='test',
        extra_params={
            'user_model': 'qwen-plus',
            'api_key': 'EMPTY',
            'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'generation_config': {
                'temperature': 0.0,
                'max_tokens': 4096,
            }
        }
    )
)
class Tau2BenchAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import(
            'tau2',
            package='git+https://github.com/sierra-research/tau2-bench@v0.2.0',
            raise_error=True,
            feature_name=self.pretty_name
        )

        # setup user model args
        self.user_model = self.extra_params.get('user_model', 'qwen-plus')
        self.api_key = self.extra_params.get('api_key', 'EMPTY')
        self.api_base = self.extra_params.get('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.generation_config = self.extra_params.get('generation_config', {'temperature': 0.0, 'max_tokens': 4096})

    def load(self):
        # Load dataset
        dataset_name_or_path = self.dataset_id
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from {dataset_name_or_path}')
            dataset_path = dataset_name_or_path
        else:
            from modelscope import dataset_snapshot_download
            logger.info(f'Loading dataset from modelscope: > dataset_name: {dataset_name_or_path}')
            dataset_path = dataset_snapshot_download(dataset_name_or_path)

        # Set Tau2 data dir
        os.environ['TAU2_DATA_DIR'] = dataset_path

        # Load data for each domain
        from tau2.agent.llm_agent import LLMGTAgent
        from tau2.registry import registry

        data_dict = defaultdict(dict)
        for domain_name in self.subset_list:
            logger.info(f'Loading Tau2-Bench environment: {domain_name}')
            # Get tasks
            task_loader = registry.get_tasks_loader(domain_name)
            tasks = task_loader()
            tasks = [task for task in tasks if LLMGTAgent.check_valid_task(task)]
            tasks = [task.model_dump(exclude_unset=True) for task in tasks]

            # load dataset
            dataset = DictDataLoader(
                dict_list=tasks,
                sample_fields=self.record_to_sample,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
            ).load()

            data_dict[domain_name] = dataset

        test_dataset = DatasetDict(data_dict)

        return test_dataset, None

    def record_to_sample(self, record: Dict) -> Sample:
        """Convert a data record to a Sample object."""
        return Sample(
            input=[ChatMessageUser(content=record['description']['purpose'] or '')],
            target='',  # Will use the record for evaluation
            subset_key=record['user_scenario']['instructions']['domain'],
            metadata=record  # Store the full record for evaluation
        )

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        from .generation import predict
        return predict(model, sample, adapter_instance=self)

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state) -> Score:

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        try:
            # Parse the prediction to get the reward
            task_result = task_state.metadata['task_result']
            reward = task_result['reward']

            score.value = {
                'Pass^1': float(reward),
            }
            score.explanation = f'Task completed with reward: {reward}'
            score.metadata = {
                'task_result': task_result,
            }
            score.main_score_name = 'Pass^1'

        except Exception as e:
            score.value = {'Pass^1': 0.0}
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata = {'error': str(e)}
            score.main_score_name = 'Pass^1'

        return score
