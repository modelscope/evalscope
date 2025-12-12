# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .utils import build_full_code, normalize_languages

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='multiple_humaneval',
        pretty_name='MultiPL-E HumanEval',
        tags=[Tags.CODING],
        description='This multilingual HumanEval was from MultiPL-E. 18 languages were implemented and tested. '
        '**Sandbox environment is needed for execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**',  # noqa: E501
        dataset_id='evalscope/MultiPL-E',
        subset_list=[
            'humaneval-cpp',
            'humaneval-ts',
            'humaneval-sh',
            'humaneval-cs',
            'humaneval-go',
            'humaneval-java',
            'humaneval-lua',
            'humaneval-js',
            'humaneval-php',
            'humaneval-pl',
            'humaneval-rkt',
            'humaneval-r',
            'humaneval-rs',
            'humaneval-scala',
            'humaneval-swift',
            'humaneval-rb',
            'humaneval-d',
            'humaneval-jl',
        ],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template='{prompt}',
        review_timeout=30,
        sandbox_config={
            'image': 'volcengine/sandbox-fusion:server-20250609',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {},
                'multi_code_executor': {}  # Multi-language code executor
            },
            'memory_limit': '2g',
            'cpu_limit': '2.0',
        },
    )
)
class MultiPLEHumanEvalAdapter(DefaultDataAdapter):
    """
    MultiPL-E HumanEval adapter using the new data processing framework.
    Assumptions:
    - Each subset is a single language suite.
    - Records contain: 'prompt', 'tests', optional 'stop_tokens', 'language', and id: 'task_id' or 'name'.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        return Sample(
            input=record['prompt'],
            target='',
            metadata={
                'tests': record['tests'],
                'stop_tokens': record.get('stop_tokens', []),
                'task_id': record.get('name', record.get('task_id')),
                'language': record.get('language'),
                'doctests': record.get('doctests', ''),
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Freeform SFT prompt:
        - Fence the given prompt with the language derived from metadata.language ("humaneval-<lang>").
        - Add a short instruction requesting full code without a Main entrypoint.
        """
        extract_lang, _ = normalize_languages(sample.metadata.get('language'))
        instruction = (
            'Please complete the above code according to the requirements in the docstring. '
            'Write the complete code and wrap it in markdown fenced code. The code should not contain `Main` function.'
        )
        return f'```{extract_lang}\n{sample.input}\n```\n\n{instruction}'

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """SFT extraction: extract fenced code, stop at language end tokens, remove entrypoints, append tests."""
        return build_full_code(
            prediction=prediction,
            language=task_state.metadata.get('language'),
            stop_tokens=task_state.metadata.get('stop_tokens', []),
            tests=task_state.metadata.get('tests', ''),
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Run code in sandbox and return pass/fail."""
        if not self.use_sandbox:
            raise RuntimeError(
                'MultiPL-E HumanEval requires sandboxed code execution for safety. Enable use_sandbox in TaskConfig.'
            )

        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        _, run_language = normalize_languages(task_state.metadata.get('language'))

        res = self.execute_code_in_sandbox(
            code=filtered_prediction,
            timeout=self.review_timeout,
            language=run_language,
        )
        passed = res.get('status') == 'success'
        score.value = {'acc': passed}
        score.metadata = {
            'task_id': task_state.metadata.get('task_id'),
            'timeout': self.review_timeout,
            'execution_result': res,
            'run_language': run_language,
        }
        score.main_score_name = 'acc'
        return score
