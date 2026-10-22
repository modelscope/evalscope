# Copyright (c) Alibaba, Inc. and its affiliates.
"""
EQ-Bench Benchmark Adapter for EvalScope

This adapter implements the EQ-Bench benchmark for evaluating language models
on emotional intelligence tasks. It uses the official EQ-Bench v2 per-question
scoring formula with the default English, non-revision dataset.

The scoring functions are imported from answer_validation.py, which is based on
the official EQ-Bench reference implementation.

References:
- Paper: https://arxiv.org/abs/2312.06281
- Homepage: https://eqbench.com/
- Official Implementation: https://github.com/EQ-bench/EQ-Bench/blob/main_v2_4/lib/scoring.py
"""

import ast
import json
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoreStatus, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """{question}"""


@register_benchmark(
    BenchmarkMeta(
        name='eq_bench',
        pretty_name='EQ-Bench',
        evaluation_version='v1.1',
        tags=[Tags.INSTRUCTION_FOLLOWING],
        description="""
## Overview

EQ-Bench is a benchmark for evaluating language models on emotional intelligence tasks. It assesses the ability to predict likely emotional responses of characters in dialogues by rating the intensity of possible emotional reactions.

## Task Description

- **Task Type**: Emotional Intelligence Assessment
- **Input**: Dialogue scenario with characters
- **Output**: Emotion intensity ratings in specific format
- **Domains**: Emotional understanding, social cognition

## Key Features

- Tests ability to predict emotional responses in conversations
- Requires rating intensity of multiple possible emotions
- Uses official EQ-Bench v2 scoring algorithm
- Scoring includes sigmoid scaling for small differences
- Adjustment constant ensures random answers score 0

## Evaluation Notes

- Default evaluation uses the **validation** split
- Primary metric: **EQ-Bench Score**, stored as the official score divided by 100 and displayed on the official scale
- Valid per-question scores can be negative (stored range: -1.9908 to 1); they are not clamped before averaging
- Unparseable or invalid answers and samples missing v2 full-scale references are excluded from the score average; the metric count is the number of scored samples
- Uses zero-shot evaluation (no few-shot examples)
- Responses must include four emotion ratings in `Emotion: score` format
- This adapter does not reproduce the official runner's format-failure resampling or optional revision mode
- Official algorithm from [Paper](https://arxiv.org/abs/2312.06281) | [Homepage](https://eqbench.com/)
""",
        dataset_id='evalscope/EQ-Bench',  # ModelScope dataset ID, falls back to local if not found
        hf_dataset_id='pbevan11/EQ-Bench',
        metric_list=['eq_bench_score'],  # Official v2 full-scale scoring
        few_shot_num=0,
        train_split=None,
        eval_split='validation',  # Will look for main_validation.csv or validation.csv
        prompt_template=PROMPT_TEMPLATE,
    )
)
class EQBenchAdapter(DefaultDataAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_reference_dict(self, value: Any) -> Dict[str, Any]:
        """
        Parse a reference answer that may be a quoted string representing a Python dict or JSON.

        Args:
            value: Raw value which can be a dict or a string containing a dict

        Returns:
            Parsed dictionary, or {} on failure
        """
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return {}

        ref_str = value.strip()
        # Strip outer quotes if present
        if (ref_str.startswith('"') and ref_str.endswith('"')) or (ref_str.startswith("'") and ref_str.endswith("'")):
            ref_str = ref_str[1:-1]

        # Try Python literal dict first, then JSON
        try:
            return ast.literal_eval(ref_str)
        except (ValueError, SyntaxError):
            try:
                return json.loads(ref_str)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f'Failed to parse reference answer: {e}')
                return {}

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record: Dictionary containing 'prompt', 'reference_answer', 'reference_answer_fullscale'

        Returns:
            Sample object with prompt as input and reference_answer as target
        """
        prompt = record.get('prompt', '').strip()
        reference_answer = record.get('reference_answer', '')
        reference_answer_fullscale = record.get('reference_answer_fullscale', '')

        # Parse the reference answer dictionaries
        try:
            ref_answer_dict = self._parse_reference_dict(reference_answer)
            ref_fullscale_dict = self._parse_reference_dict(reference_answer_fullscale)
        except Exception as e:
            logger.warning(f'Failed to parse reference answer: {e}')
            ref_answer_dict = {}
            ref_fullscale_dict = {}

        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target=str(reference_answer),  # Store as string for comparison
            metadata={
                'reference_answer': ref_answer_dict,
                'reference_answer_fullscale': ref_fullscale_dict,
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores using official EQ-Bench v2 scoring algorithm.

        Uses sigmoid scaling for small differences (≤5) and linear scaling for large differences (>5),
        with an adjustment constant (0.7477) that makes random answers score 0.

        Returns the official score divided by 100, preserving negative scores.
        Invalid answers or missing v2 references are excluded from aggregation.
        """
        # Import official EQ-Bench evaluation functions from bundled implementation
        # The answer_validation.py module is bundled with this package to ensure
        # it's available both in development (pip install -e .) and production (pip install .) installations.
        from evalscope.benchmarks.eq_bench.answer_validation import (
            calculate_score_fullscale,
            parse_answers,
            validate_answer_format,
        )

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            main_score_name='eq_bench_score',
            status=ScoreStatus.EXCLUDED,
            metadata={'metric_unavailable': True},
        )

        try:
            # A v1 reference cannot be scored with the v2 full-scale formula.
            reference_to_use = task_state.metadata.get('reference_answer_fullscale', {})

            if not reference_to_use:
                logger.warning('No v2 full-scale reference answer found in metadata')
                score.explanation = 'No v2 full-scale reference answer available'
                return score

            # Parse the prediction using official parser
            # This extracts {emotion_name: score} format from the model output
            first_pass_answers, _ = parse_answers(filtered_prediction, REVISE=False)

            if not first_pass_answers:
                logger.warning('Failed to parse any emotion scores from prediction')
                score.explanation = 'Failed to parse prediction'
                return score

            # Get reference emotion names for validation
            reference_emotions = [reference_to_use.get(f'emotion{i}', '') for i in range(1, 5)]

            # Validate answer format using official validator
            is_valid, error_msg = validate_answer_format(first_pass_answers, reference_emotions)

            if not is_valid:
                logger.warning(f'Invalid answer format: {error_msg}')
                logger.debug(f'Parsed answers: {first_pass_answers}')
                logger.debug(f'Expected emotions: {reference_emotions}')
                score.explanation = f'Invalid format: {error_msg}'
                return score

            # Calculate score using official v2 full-scale scoring algorithm
            eq_score = calculate_score_fullscale(reference_to_use, first_pass_answers)

            if eq_score is None:
                logger.warning('calculate_score_fullscale returned None')
                score.explanation = 'Scoring failed'
                return score

            # Divide the raw score by 10 for storage. Keep negative scores: the official
            # benchmark averages them, which is what makes random answers score 0.
            normalized_score = eq_score / 10.0

            score.value = {'eq_bench_score': normalized_score}
            score.status = ScoreStatus.SUCCESS
            score.metadata.pop('metric_unavailable')
            score.explanation = f'EQ-Bench Score: {normalized_score:.2f} (raw: {eq_score:.3f}/10)'

            logger.debug(f'Prediction: {first_pass_answers}')
            logger.debug(f'Reference: {reference_emotions}')
            logger.debug(f'Score: {normalized_score:.2f}')

        except Exception as e:
            logger.error(f'Error calculating EQ-Bench score: {e}')
            import traceback

            logger.debug(traceback.format_exc())
            score.explanation = f'Evaluation error: {str(e)}'

        return score
