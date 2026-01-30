# Copyright (c) Alibaba, Inc. and its affiliates.

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
        description="""
## Overview

HumanEval is a benchmark for evaluating the code generation capabilities of language models. It consists of 164 hand-written Python programming problems with function signatures, docstrings, and comprehensive test cases.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Function signature with docstring describing the expected behavior
- **Output**: Complete Python function implementation
- **Languages**: Python only

## Key Features

- 164 hand-crafted programming problems
- Each problem includes a function signature, docstring, and test cases
- Problems range from simple string manipulation to complex algorithms
- Canonical solutions provided for reference
- Automatic correctness verification through test execution

## Evaluation Notes

- **Security Warning**: By default, code is executed in the local environment. We strongly recommend using sandbox execution for safety. See the [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for details.
- Supports `pass@k` metric calculation for measuring generation quality
- Default timeout is 4 seconds per problem
- Code is extracted from markdown code blocks if present
""",
        dataset_id='opencompass/humaneval',
        subset_list=['openai_humaneval'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=
        'Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{question}',  # noqa: E501
        review_timeout=4,
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {}
            }
        },
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
        score.value = {'acc': passed}
        score.metadata = {'task_id': problem['task_id'], 'timeout': self.review_timeout, 'execution_result': res}
        score.main_score_name = 'acc'

        return score
