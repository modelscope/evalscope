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

FEWSHOT_PROMPT = """You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n[BEGIN]\ndef similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)\n[DONE]
You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True\n[BEGIN]\nimport math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result\n[DONE]
You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]\n[BEGIN]\nimport heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums\n[DONE]
You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}
[BEGIN]
"""  # noqa: E501

PROMPT = """You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='mbpp',
        pretty_name='MBPP',
        tags=[Tags.CODING],
        description='MBPP (Mostly Basic Python Problems Dataset): The benchmark consists of around 1,000 '
        'crowd-sourced Python programming problems, designed to be solvable by entry level programmers, '
        'covering programming fundamentals, standard library functionality, and so on. Each problem '
        'consists of a task description, code solution and 3 automated test cases.'
        '**Sandbox environment is needed for execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**',  # noqa: E501
        dataset_id='google-research-datasets/mbpp',
        subset_list=['full'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        train_split='prompt',
        eval_split='test',
        few_shot_num=3,
        prompt_template=PROMPT,
        few_shot_prompt_template=FEWSHOT_PROMPT,
        review_timeout=20,
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {}
            }
        },
    )
)
class MBPPAdapter(DefaultDataAdapter):
    """
    MBPP adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""

        return Sample(
            input=record['text'],
            target=record['code'],
            metadata={
                'test_list': record['test_list'],
                'task_id': record['task_id'],
                'test_setup_code': record['test_setup_code'],
            }
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return ''

    def format_prompt_template(self, sample: Sample) -> str:
        tests = '\n'.join(sample.metadata['test_list'])
        return self.prompt_template.format(question=sample.input, tests=tests)

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        tests = '\n'.join(sample.metadata['test_list'])
        return self.few_shot_prompt_template.format(question=sample.input, tests=tests)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""

        code = self.postprocess_completion(prediction)
        code = '\n'.join([task_state.metadata['test_setup_code'], code])
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
