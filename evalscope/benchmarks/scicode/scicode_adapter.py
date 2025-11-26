# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import os
from typing import Any, Dict, List, cast
from pathlib import Path
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.metric.scorer import AggScore, SampleScore
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .prompt_templates import (
    INITIAL_PROMPT,
    INITIAL_PROMPT_PROVIDE_BACKGROUND,
    SUBPROBLEM_PROMPT,
    SUBPROBLEM_PROMPT_PROVIDE_BACKGROUND,
)

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='scicode',
        pretty_name='SciCode',
        tags=[Tags.CODING],
        description='SciCode is a challenging benchmark designed to evaluate the capabilities of '
        'language models (LMs) in generating code for solving realistic scientific '
        'research problems. It has a diverse coverage of 16 subdomains from 5 domains: '
        'Physics, Math, Material Science, Biology, and Chemistry. Unlike previous benchmarks '
        'that consist of exam-like question-answer pairs, SciCode is converted from real research '
        'problems. SciCode problems naturally factorize into multiple subproblems, each involving '
        'knowledge recall, reasoning, and code synthesis. '
        '**By default the code is executed in local environment. We recommend using sandbox execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**',  # noqa: E501
        dataset_id='evalscope/SciCode',
        metric_list=['main_problem_pass_rate', 'subproblem_pass_rate'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        review_timeout=300,
        prompt_template=INITIAL_PROMPT,
        extra_params={
            'provide_background': {
                'type':
                'bool',
                'value':
                False,
                'description':
                'Include scientific background information written by scientists for the problem in the model\'s prompt.'
            }
        }
    )
)
class SciCodeAdapter(DefaultDataAdapter):
    """
    SciCode adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.provide_background = self.extra_params.get('provide_background', False)
        self.system_prompt = INITIAL_PROMPT_PROVIDE_BACKGROUND if self.provide_background else INITIAL_PROMPT
        self.prompt_template = SUBPROBLEM_PROMPT_PROVIDE_BACKGROUND if self.provide_background else SUBPROBLEM_PROMPT
        self.docker_path = Path(__file__).parent / 'docker'
        self._use_custom_image = True

    def load(self):
        test_data_path = self.docker_path / "test_data.h5"
        if not test_data_path.exists():
            data_url = 'https://modelscope.cn/datasets/evalscope/SciCode/resolve/master/test_data.h5'
            logger.info(f'Downloading test data for SciCode benchmark from {data_url}...')
            os.system(f'wget --tries=3 -O {test_data_path} {data_url}')
        return super().load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=[ChatMessageUser(content=record['problem_id'])], 
            metadata=record
        )

    def run_inference(self, model, sample, output_dir, **kwargs) -> TaskState:
        system_prompt = self.system_prompt.format(required_dependencies=sample.metadata['required_dependencies'])
        messages = [ChatMessageSystem(content=system_prompt)]

        for subproblem in sample.metadata['sub_steps']:
            formatted_prompt = self.prompt_template.format(**subproblem)
            chat_message = ChatMessageUser(content=formatted_prompt)
            messages.append(chat_message)
            logger.info(f'Submitting subproblem {subproblem["step_number"]} to model...')
            model_output = model.generate(input=messages)
            messages.append(model_output.message)

        return TaskState(
            model=model.name,
            sample=sample,
            messages=messages,
            completed=True,
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        subproblem_scorers = [
            self._verify_subproblem(subproblem=subproblem, state=task_state)
            for subproblem in task_state.metadata['sub_steps']
        ]
        def reduce(scores: List[Score]) -> Score:
            result = {
                k: v
                for score in scores
                for k, v in cast(dict[str, Any], score.value).items()
            }

            explanations = "\n".join(
                [score.explanation for score in scores if score.explanation is not None]
            )
            
            main_problem = all(
                value == 1
                for value in cast(dict[str, Any], result).values()
            )
            score = Score(
                value={
                    'main_problem': 1.0 if main_problem else 0.0,
                    **result,
                },
                explanation=explanations,
                main_score_name='main_problem',
            )
            return score
        
        return reduce(subproblem_scorers)

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        subproblem_solved = 0
        subproblem_count = 0
        main_problem_solved = 0
        main_problem_count = len(sample_scores)
        for sample_score in sample_scores:
            for k, v in sample_score.score.value.items():
                if k == 'main_problem':
                    if v == 1.0:
                        main_problem_solved += 1
                else:
                    subproblem_count += 1
                    if v == 1.0:
                        subproblem_solved += 1
        agg_scores = [
            AggScore(
                metric_name='main_problem_pass_rate',
                score=main_problem_solved / main_problem_count if main_problem_count > 0 else 0.0,
                num=main_problem_count,
            ),
            AggScore(
                metric_name='subproblem_pass_rate',
                score=subproblem_solved / subproblem_count if subproblem_count > 0 else 0.0,
                num=subproblem_count,
            ),
        ]
        return agg_scores

    def get_build_context(self):
        return self.docker_path.as_posix(), (self.docker_path / "Dockerfile").as_posix()

    def _verify_subproblem(self, subproblem: Dict[str, Any], state: TaskState) -> bool:
        """
        Verify if a subproblem is correct.

        A subproblem is considered correct if all its test cases pass i.e. execute without erroring.
        """
        from .util import get_generated_code, subproblem_str_to_int

        def compose_code() -> str:
            """
            Compose code to be executed.

            This consists of:
                * dependencies, as defined for the main problem in state.metadata["dependencies"]
                * functions definitions produced by each previous subproblems
                * the code produced in response to **this** subproblem
                * the tests associated with **this** subproblem

            """
            subproblem_no = subproblem['step_number']
            test_cases = subproblem['test_cases']

            # Collect dependencies and the model's subproblem solutions.
            code_sections: List[str] = []
            code_sections.append(state.metadata['required_dependencies'])
            
            code_sections.append('from test_util import are_dicts_close, cmp_tuple_or_list')
            code_sections += get_generated_code(state)[:subproblem_str_to_int(subproblem_no)]

            # Collect test cases and data.
            code_sections.append('# Test cases:')
            code_sections.append('from process_data import process_hdf5_to_tuple')
            code_sections.append(f"targets = process_hdf5_to_tuple('{subproblem_no}', {len(test_cases)})")

            for i, test_case in enumerate(test_cases):
                code_sections.append(f'target = targets[{i}]')
                code_sections.append(test_case)

            # Compose code sections to a single string.
            code = '\n'.join(code_sections)
            return code

        def format_explanation(code: str, result: Dict) -> str:
            """Format the score explanation, based on the result of code execution."""
            subproblem_no = subproblem['step_number']
            explanation = (f'The following code was executed:\n\n```python\n{code}\n```\n')
            if result['status'] == 'success':
                explanation += (f'All test cases passed for subproblem {subproblem_no}.\n')
            else:
                explanation += 'Code did not pass all test cases.\n'
                if result.get('stderr'):
                    explanation += f"Error details:\n```python\n{result['stderr']}\n```\n"
            return explanation

        code = compose_code()
        try:
            result = self.execute_code_in_sandbox(
                code=f"python -c \"{code}\"",
                language='shell',
                timeout=self.review_timeout,
            )
            explanation = format_explanation(code, result)

            return Score(
                value={subproblem['step_number']: 1.0 if result['status'] == 'success' else 0.0},
                prediction=code,
                explanation=explanation,
            )
        except Exception as e:
            return Score(
                value={subproblem['step_number']: 0.0},
                prediction=code,
                explanation=f'An error occurred during code execution: {e}',
            )
