# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# Example:
# {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}  # noqa


@register_benchmark(
    BenchmarkMeta(
        name='humaneval',
        pretty_name='HumanEval',
        tags=[Tags.CODING],
        description=
        'HumanEval is a benchmark for evaluating the ability of code generation models to write Python functions based on given specifications. It consists of programming tasks with a defined input-output behavior.',
        dataset_id='opencompass/humaneval',
        subset_list=['openai_humaneval'],
        metric_list=['Pass@1'],
        eval_split='test',
        prompt_template=
        'Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{question}',
        extra_params={
            'num_workers': 4,
            'timeout': 4
        },
    )
)
class HumanevalAdapter(DefaultDataAdapter):
    """
    HumanEval adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        try:
            from human_eval.data import stream_jsonl, write_jsonl
            from human_eval.evaluation import check_correctness
        except ImportError:
            raise ImportError(
                'Please install human_eval:'
                'https://github.com/openai/human-eval/tree/master#installation , '
                'Note that you need to enable the execution code in the human_eval/execution.py first.'
            )
        super().__init__(**kwargs)

        extra_params = kwargs.get('extra_params', {})
        self.k = [1]
        self.num_workers = extra_params.get('num_workers', 4)
        self.timeout = extra_params.get('timeout', 4)

        self.read_problems_func = stream_jsonl
        self.write_jsonl_func = write_jsonl
        self.eval_func = check_correctness

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        query = record['prompt']
        full_prompt = self.prompt_template.format(question=query)

        return Sample(
            input=[ChatMessageUser(content=full_prompt)],
            target=record['canonical_solution'],
            metadata={
                'task_id': record['task_id'],
                'entry_point': record['entry_point'],
                'prompt': record['prompt'],
                'test': record['test'],
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""
        return self._postprocess(prediction)

    @classmethod
    def _postprocess(cls, text: str) -> str:
        """Extract code from markdown code blocks."""
        blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
        if len(blocks) >= 1:
            text = blocks[0]
        return text

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Execute the code and check correctness
        res = self.eval_func(task_state.metadata, filtered_prediction, self.timeout)
        passed = res['passed']

        score.value = {'pass': passed}
        score.explanation = res.get('result', 'Code execution completed')
        score.metadata = {'task_id': task_state.metadata['task_id'], 'timeout': self.timeout, 'execution_result': res}
        score.main_score_name = 'pass'

        return score

    def aggregate_scores(self, sample_scores):
        from evalscope.metrics.metric import PassAtK

        # caculate pass@k here
        agg_list = []
        for metric in self.metric_list:
            if metric.lower().startswith('pass@'):
                k = int(metric.split('@')[1])
                # Get the scores for this metric
                agg = PassAtK(k)
                agg_list.extend(agg(sample_scores))
        return agg_list
