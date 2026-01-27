from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='ifeval',
        pretty_name='IFEval',
        description="""
## Overview

IFEval (Instruction-Following Eval) is a benchmark for evaluating how well language models follow explicit, verifiable instructions. It contains prompts with specific formatting, content, or structural requirements that can be objectively verified.

## Task Description

- **Task Type**: Instruction Following Evaluation
- **Input**: Prompts with explicit, verifiable constraints
- **Output**: Response that follows all specified instructions
- **Constraint Types**: Format, length, keywords, structure, etc.

## Key Features

- ~500 prompts with 25 types of verifiable instructions
- Instructions are objectively checkable (not subjective)
- Examples: "write exactly 3 paragraphs", "include the word X", "use bullet points"
- Tests instruction comprehension and compliance
- No ambiguity in evaluation criteria

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Four metrics available:
  - `prompt_level_strict`: All instructions in prompt must be followed
  - `prompt_level_loose`: Some tolerance for minor deviations
  - `inst_level_strict`: Per-instruction accuracy (strict)
  - `inst_level_loose`: Per-instruction accuracy (loose)
- `prompt_level_strict` is the primary metric
- Automatic verification of instruction compliance
""",
        tags=[Tags.INSTRUCTION_FOLLOWING],
        dataset_id='opencompass/ifeval',
        subset_list=['default'],
        metric_list=[
            'prompt_level_strict',
            'inst_level_strict',
            'prompt_level_loose',
            'inst_level_loose',
        ],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='',
    )
)
class IFEvalAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        prompt = record.get('prompt', '')
        message_list = [ChatMessageUser(content=prompt)]

        return Sample(input=message_list, target='', metadata=record)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: Dict, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        from evalscope.benchmarks.ifeval.utils import process_results

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        doc = task_state.metadata
        try:
            # Process results using the existing ifeval utility
            results = process_results(doc, [filtered_prediction])
            score.value.update(results)

            # Set main score name
            score.main_score_name = 'prompt_level_strict'

        except Exception as e:
            logger.error(f'Error calculating ifeval metrics: {e}')
            score.value = {}

        return score
