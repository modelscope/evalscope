import json
from typing import Any, Dict

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessage, dict_to_chat_message
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='tool_bench',
        pretty_name='ToolBench-Static',
        tags=[Tags.REASONING, Tags.FUNCTION_CALLING],
        description='ToolBench is a benchmark for evaluating AI models on tool use tasks. '
        'It includes various subsets such as in-domain and out-of-domain, '
        'each with its own set of problems that require step-by-step reasoning to arrive at the correct answer. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html)',
        dataset_id='AI-ModelScope/ToolBench-Static',
        subset_list=['in_domain', 'out_of_domain'],
        metric_list=['Act.EM', 'Plan.EM', 'F1', 'HalluRate', 'Rouge-L'],
        eval_split='test',
    )
)
class ToolBenchAdapter(AgentAdapter):
    """
    ToolBench adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        messages = record['messages']

        # Process messages and remove the name field, convert function messages
        processed_messages = []
        for message in messages:
            msg_dict = message.copy()
            if 'name' in msg_dict:
                del msg_dict['name']
            if 'role' in msg_dict:
                if msg_dict['role'] == 'function':
                    content = json.dumps(msg_dict, ensure_ascii=False)
                    msg_dict['role'] = 'user'
                    msg_dict['content'] = content

            # Convert to ChatMessage object
            chat_msg = dict_to_chat_message(msg_dict)
            processed_messages.append(chat_msg)

        return Sample(
            input=processed_messages,
            target='',  # Store the full record as target for evaluation
            metadata={
                'target': record['target'],
                'tools': record['tools'],
                'messages': record['messages']
            }
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from .utils import calculate_metrics

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        doc = task_state.metadata

        try:
            data = {
                'target': doc['target'],
                'predictions': filtered_prediction,
                'tools': doc['tools'],
            }
            metrics = calculate_metrics(data)

            score.value = metrics
            score.explanation = f'Metrics: {metrics}'
            score.metadata = {'target': doc['target'], 'tools': doc['tools'], 'detailed_metrics': metrics}
            # Set the main score (you can choose the most important metric)
            score.main_score_name = 'F1'

        except Exception as e:
            # Handle evaluation errors
            score.value = {'Act.EM': 0.0, 'Plan.EM': 0.0, 'F1': 0.0, 'HalluRate': 1.0, 'Rouge-L': 0.0}
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata = {'error': str(e)}
            score.main_score_name = 'F1'

        return score
