# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT = """You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='mbppplus',
        pretty_name='MBPPplus',
        tags=[Tags.CODING],
        description='This dataset includes multiple features related to programming tasks',
        dataset_id='evalscope/mbppplus',
        subset_list=['default'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=PROMPT,
        review_timeout=20,
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {}
            }
        }
    )
)
class MBPPplusAdapter(DefaultDataAdapter):
    """
    MBPPplus adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""

        return Sample(
            input=record['prompt'],
            target=record['code'],
            metadata={
                'task_id': record['task_id'],
                'source_file': record['source_file'],
                'test_imports': record['test_imports'],
                'test_list': record['test_list'],
                'test': record['test'],
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        tests = '\n'.join(sample.metadata['test_list'])
        return self.prompt_template.format(question=sample.input, tests=tests)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""
        code = self.postprocess_completion(prediction)
        if 'if __name__ ==' in code:
            code = code[:code.index('if __name__ ==')]
        return code

    @classmethod
    def postprocess_completion(cls, completion, stop_words=['\nassert', '\n"""']):
        from evalscope.utils.code_utils import extract_code_from_freeform_completion

        if '[DONE]' in completion:
            completion = completion[:completion.index('[DONE]')]

        code, _ = extract_code_from_freeform_completion(completion, 'python', first_block_only=True)

        for st in stop_words:
            index = code.find(st)
            if index != -1:
                code = code[:index]
        return code

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:

        if not self.use_sandbox:
            raise RuntimeError(
                f'{self.pretty_name} benchmark requires sandboxed code '
                'execution for safety reasons. Please set use_sandbox in the task configuration.'
            )

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        problem = task_state.metadata
        completion = filtered_prediction
        # Append test cases to the completion
        for test in task_state.metadata['test_list']:
            completion += '\n' + test + '\n'

        res = self.execute_code_in_sandbox(code=completion, timeout=self.review_timeout, language='python')
        passed = res.get('status') == 'success'
        # Set score values
        score.value = {'acc': passed}
        score.metadata = {'task_id': problem['task_id'], 'timeout': self.review_timeout, 'execution_result': res}
        score.main_score_name = 'acc'

        return score
