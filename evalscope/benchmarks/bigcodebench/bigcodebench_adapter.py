# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import DockerImageBuilder, DockerImageSpec
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = """
## Overview

BigCodeBench is an easy-to-use benchmark for solving practical and challenging tasks via code. It evaluates the true programming capabilities of large language models (LLMs) in a more realistic setting with diverse function calls from 139 popular libraries covering 723 API calls.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Programming task description (docstring or natural language instruction)
- **Output**: Complete Python function implementation
- **Libraries**: 139 popular Python libraries (numpy, pandas, sklearn, etc.)

## Key Features

- 1,140 rich-context programming tasks in Python
- Two evaluation modes: Complete (docstring) and Instruct (natural language)
- Covers diverse function calls from 139 popular libraries
- Uses unittest.TestCase for thorough correctness verification
- Supports pass@k metric calculation

## Evaluation Notes

- **Sandbox Required**: Requires sandbox environment with 70+ Python libraries pre-installed
- Two modes available via `split` parameter: `complete` (docstring completion) or `instruct` (NL instruction)
- Default timeout is 240 seconds per problem
- `calibrate` option prepends code_prompt to align function signatures
- See [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for setup
"""  # noqa: E501

HARD_DESCRIPTION = """
## Overview

BigCodeBench-Hard is a curated subset of BigCodeBench containing 148 tasks that are more aligned with real-world programming tasks. These tasks require more complex reasoning and multi-step problem solving.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Programming task description (docstring or natural language instruction)
- **Output**: Complete Python function implementation
- **Difficulty**: Higher than BigCodeBench-Full, closer to real-world complexity

## Key Features

- 148 challenging tasks selected from BigCodeBench
- Requires complex reasoning and multi-tool usage
- Two evaluation modes: Complete (docstring) and Instruct (natural language)
- Uses unittest.TestCase for thorough correctness verification
- Supports pass@k metric calculation

## Evaluation Notes

- **Sandbox Required**: Requires sandbox environment with 70+ Python libraries pre-installed
- Two modes available via `split` parameter: `complete` (docstring completion) or `instruct` (NL instruction)
- Default timeout is 240 seconds per problem
- `calibrate` option prepends code_prompt to align function signatures
- See [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for setup
"""  # noqa: E501

_EXTRA_PARAMS = {
    'split': {
        'type': 'str',
        'description': 'Evaluation mode: "complete" (docstring completion) or "instruct" (NL instruction).',
        'value': 'instruct',
        'choices': ['complete', 'instruct']
    },
    'version': {
        'type': 'str',
        'description': 'Dataset version. Use "default" for the latest available version.',
        'value': 'default'
    },
    'calibrate': {
        'type': 'bool',
        'description': 'Whether to prepend code_prompt to the solution for function signature alignment.',
        'value': True
    },
    'docker_build_context': {
        'type': 'str',
        'description': 'Optional local Docker build context. When set, overrides the default sandbox image.',
        'value': ''
    },
    'dockerfile': {
        'type': 'str',
        'description': 'Dockerfile path inside docker_build_context.',
        'value': 'Dockerfile'
    },
    'force_rebuild': {
        'type': 'bool',
        'description': 'Force rebuilding the optional local Docker image.',
        'value': False
    }
}

_SANDBOX_CONFIG = {
    'image': 'bigcodebench/bigcodebench-evaluate:latest',
    'tools_config': {
        'shell_executor': {},
        'python_executor': {}
    },
    'memory_limit': '4g',
}


@register_benchmark(
    BenchmarkMeta(
        name='bigcodebench',
        pretty_name='BigCodeBench',
        tags=[Tags.CODING],
        description=DESCRIPTION,
        dataset_id='evalscope/bigcodebench',
        subset_list=['default'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='v0.1.4',
        prompt_template='{prompt}',
        review_timeout=240,
        extra_params=_EXTRA_PARAMS,
        sandbox_config=_SANDBOX_CONFIG,
    )
)
class BigCodeBenchAdapter(DefaultDataAdapter):
    """
    BigCodeBench adapter for evaluating code generation with diverse function calls.
    Supports both 'complete' (docstring-based) and 'instruct' (NL-based) evaluation modes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._split = self.extra_params.get('split', 'instruct')
        self._calibrate = self.extra_params.get('calibrate', True)
        self._maybe_prepare_local_image()
        # Dataset uses version-based splits (e.g. v0.1.4), override eval_split if version is specified
        version = self.extra_params.get('version', 'default')
        if version != 'default':
            self.eval_split = version

    def _maybe_prepare_local_image(self) -> None:
        build_context = self.extra_params.get('docker_build_context') or ''
        if not build_context:
            return
        result = DockerImageBuilder().build_or_reuse(
            DockerImageSpec(
                name_prefix=f'evalscope-{self.name}',
                context_dir=build_context,
                dockerfile=self.extra_params.get('dockerfile') or 'Dockerfile',
                cache_key_parts=[self.name, 'bigcodebench'],
                force_rebuild=bool(self.extra_params.get('force_rebuild', False)),
            )
        )
        self._benchmark_meta.sandbox_config = dict(self._benchmark_meta.sandbox_config or {})
        self._benchmark_meta.sandbox_config['image'] = result.image_tag
        logger.info(f'{self.pretty_name} using local Docker image: {result.image_tag} (reused={result.reused})')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        prompt = record[f'{self._split}_prompt']

        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target=record['canonical_solution'],
            metadata={
                'task_id': record['task_id'],
                'entry_point': record['entry_point'],
                'complete_prompt': record['complete_prompt'],
                'code_prompt': record['code_prompt'],
                'test': record['test'],
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""
        from evalscope.utils.code_utils import extract_code_from_freeform_completion

        code, _ = extract_code_from_freeform_completion(prediction, 'python', first_block_only=True)
        if not code.strip():
            # Fallback: return prediction as-is if extraction fails
            code = prediction
        return code

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Evaluate the generated code by running unittest in sandbox."""
        if not self.use_sandbox:
            raise RuntimeError(
                f'{self.pretty_name} benchmark requires sandboxed code execution '
                'due to its 70+ library dependencies. Please enable sandbox in the task configuration.'
            )

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        problem = task_state.metadata
        solution = filtered_prediction

        # Calibrate: prepend code_prompt to ensure function signature alignment
        if self._calibrate:
            solution = problem['code_prompt'] + '\n    pass\n' + solution

        # Construct full test program: solution + unittest test class
        check_program = solution + '\n' + problem['test']

        # Execute in sandbox
        res = self.execute_code_in_sandbox(code=check_program, timeout=self.review_timeout, language='python')
        passed = res.get('status') == 'success'

        # Set score values
        score.value = {'acc': passed}
        score.metadata = {
            'task_id': problem['task_id'],
            'timeout': self.review_timeout,
            'execution_result': res,
        }
        score.main_score_name = 'acc'

        return score


@register_benchmark(
    BenchmarkMeta(
        name='bigcodebench_hard',
        pretty_name='BigCodeBench-Hard',
        tags=[Tags.CODING],
        description=HARD_DESCRIPTION,
        dataset_id='evalscope/bigcodebench-hard',
        subset_list=['default'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='v0.1.4',
        prompt_template='{prompt}',
        review_timeout=240,
        extra_params=_EXTRA_PARAMS,
        sandbox_config=_SANDBOX_CONFIG,
    )
)
class BigCodeBenchHardAdapter(BigCodeBenchAdapter):
    """
    BigCodeBench-Hard adapter. Inherits all logic from BigCodeBenchAdapter.
    Uses the curated hard subset (148 tasks) of BigCodeBench.
    """
    pass
