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
        review_timeout=4,
    )
)
class HumanevalAdapter(DefaultDataAdapter):
    """
    HumanEval adapter using the new data processing framework.
    """

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
        problem = task_state.metadata
        completion = filtered_prediction

        if not self.use_sandbox:
            from .utils import check_correctness

            # Execute the code and check correctness
            res = check_correctness(problem=problem, completion=completion, timeout=self.review_timeout)
            passed = res['passed']
        else:
            check_program = (
                problem['prompt'] + completion + '\n' + problem['test'] + '\n' + f"check({problem['entry_point']})"
            )
            res = self.execute_code_in_sandbox(code=check_program, timeout=self.review_timeout, language='python')
            passed = res.get('status') == 'success'
        # Set score values
        score.value = {'pass': passed}
        score.metadata = {'task_id': problem['task_id'], 'timeout': self.review_timeout, 'execution_result': res}
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
