# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from pathlib import Path
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

DESCRIPTION = """HumanEval Plus is a rigorous extension of OpenAI's original HumanEval dataset, designed to address the high false-positive rates found in code generation benchmarks. While the original HumanEval uses a limited number of test cases per problem, HumanEval+ augments these with tens of thousands of automatically generated inputs (using LLM-based and mutation-based strategies). This massively expanded test suite exposes edge-case bugs and functional errors that previously went undetected, making it a much stricter and more accurate standard for evaluating the correctness of AI-generated code.
**By default the code is executed in local environment. We recommend using sandbox execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**'
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='humaneval_plus',
        pretty_name='HumanEvalPlus',
        tags=[Tags.CODING],
        description=DESCRIPTION,
        dataset_id='evalscope/humanevalplus',
        subset_list=['default'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=
        'Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{question}',  # noqa: E501
        review_timeout=300,
        sandbox_config={
            'image': 'python3.11-numpy',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {}
            }
        },
    )
)
class HumanevalplusAdapter(DefaultDataAdapter):
    """
    HumanEvalPlus adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docker_path = Path(__file__).parent / 'docker'
        self._use_custom_image = True

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

    def get_build_context(self):
        """
        Get the build context for the docker image.
        """
        return self.docker_path.as_posix(), (self.docker_path / 'Dockerfile').as_posix()

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
