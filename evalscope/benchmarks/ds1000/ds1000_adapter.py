# DS-1000 adapter: data science code generation across seven Python libraries, scored by execution.
#
# Paper: https://arxiv.org/abs/2211.11501 (ICML 2023)
# Dataset: https://huggingface.co/datasets/xlangai/DS-1000 (CC BY-SA 4.0)
# Official evaluation: https://github.com/xlang-ai/DS-1000

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.mixin import CodeExecutionSandboxMixin
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import build_test_program, extract_code, run_locally

SUBSET_LIST = ['Pandas', 'Numpy', 'Matplotlib', 'Sklearn', 'Scipy', 'Pytorch', 'Tensorflow']

# System prompt used by the official chat-model inference script (run_openai.py).
SYSTEM_PROMPT = (
    'Write a short code following the given format and indentation. Place the executable code between '
    '<code> and </code> tags, without any other non-executable things.'
)

DESCRIPTION = """
## Overview

DS-1000 is a code generation benchmark of 1,000 data science problems collected from StackOverflow,
covering seven Python libraries. Each problem gives the context code and asks for the missing part;
the answer is checked by running it against the problem's test cases.

## Task Description

- **Task Type**: Code completion for data science
- **Input**: A StackOverflow-style question with context code, ending with `BEGIN SOLUTION <code>`
- **Output**: The code that fills the gap (in `<code>` tags or a fenced block)
- **Libraries**: Pandas (291), NumPy (220), Matplotlib (155), Scikit-learn (115), SciPy (106),
  PyTorch (68), TensorFlow (45)

## Key Features

- Problems are perturbed to reduce memorization: 452 origin, 152 surface, 234 semantic and 162
  difficult-rewrite problems (recorded as `perturbation_type`)
- Execution-based scoring with the official test harness carried in each problem (`code_context`),
  including surface-form checks (`test_string`) that, e.g., forbid loops where vectorization is asked

## Evaluation Notes

- `accuracy` (primary): the problem passes when the official test program
  (`code_context` + `test_execution(code)` [+ `test_string(code)`]) runs without error
- Uses the system prompt of the official chat-model script; code is taken after an opening `<code>` tag
  or from a fenced block, up to `</code>` / `END SOLUTION`
- **Environment matters**: tests compare against outputs of specific library versions. The official
  environment is Python 3.10 with pandas 1.5.3, numpy 1.26.4, scipy 1.12.0, scikit-learn 1.4.0,
  matplotlib 3.8.4, torch 2.2.0 and tensorflow-cpu 2.16.1 (see `environment.yml` in the dataset)
- By default each test program runs in a separate local Python process (timeout `review_timeout`,
  120 s as in the official harness), which is **not a security sandbox**; enable the sandbox with an
  image that has the official environment to isolate execution
- Subsets are the libraries; `limit` applies per library
- [Paper](https://arxiv.org/abs/2211.11501) | [Code](https://github.com/xlang-ai/DS-1000) |
  [Dataset](https://huggingface.co/datasets/xlangai/DS-1000)
"""


@register_benchmark(
    BenchmarkMeta(
        name='ds1000',
        pretty_name='DS-1000',
        dataset_id='xlangai/DS-1000',
        hf_dataset_id='xlangai/DS-1000',
        paper_url='https://arxiv.org/abs/2211.11501',
        tags=[Tags.CODING],
        description=DESCRIPTION,
        subset_list=SUBSET_LIST,
        eval_split='test',
        few_shot_mode='disabled',
        metric_list=['accuracy'],
        system_prompt=SYSTEM_PROMPT,
        prompt_template='{question}',
        review_timeout=120,
        sandbox_config={'image': 'python:3.10-slim', 'tools_config': {'python_executor': {}}},
        evaluation_version='v1.0',
    )
)
class DS1000Adapter(CodeExecutionSandboxMixin, DefaultDataAdapter):
    """Data adapter for DS-1000, scored by running the official per-problem test program."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # One split holds every problem; split it into one subset per library.
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        meta = record['metadata']
        return Sample(
            input=[ChatMessageUser(content=record['prompt'])],
            target=record['reference_code'],
            subset_key=meta['library'],
            metadata={
                'problem_id': meta['problem_id'],
                'library': meta['library'],
                'perturbation_type': meta['perturbation_type'],
                'code_context': record['code_context'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return extract_code(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        program = build_test_program(task_state.metadata['code_context'], filtered_prediction)
        if self.use_sandbox:
            res = self.execute_code_in_sandbox(code=program, timeout=self.review_timeout, language='python')
            passed = res.get('status') == 'success'
        else:
            res = run_locally(program, timeout=self.review_timeout)
            passed = res['passed']
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'accuracy': passed},
            main_score_name='accuracy',
            metadata={
                'problem_id': task_state.metadata['problem_id'],
                'perturbation_type': task_state.metadata['perturbation_type'],
                'execution_result': res,
            },
        )
