# flake8: noqa: E501
import json
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = """
## Overview

MeasureBench is a comprehensive benchmark for evaluating the ability of vision-language models (VLMs) \
to read values from measuring instruments. It covers both **real-world photographs** and \
**synthetically generated images** of 26 instrument types across 4 design categories.

## Task Description

- **Task Type**: Free-form Visual Question Answering (instrument reading)
- **Input**: An image of a measuring instrument + a reading question
- **Output**: The instrument's current reading (numeric value or time, with unit)
- **Domains**: Ammeters, clocks, thermometers, scales, speedometers, and 21 more instrument types

## Key Features

- 2,442 total samples across two splits: real_world (1,272) and synthetic_test (1,170)
- 26 instrument types, 4 design categories (dial, digital, analog, linear)
- Accepts a tolerance interval around the correct value rather than requiring an exact match
- For clocks: handles both 12-hour and 24-hour ambiguity via multiple valid intervals
- Unit recognition is evaluated separately from numeric accuracy

## Evaluation Notes

- Default splits: **real_world** and **synthetic_test** (treated as separate subsets)
- Primary metric: **Accuracy** (acc) — ``all_correct``: number *and* unit both correct
- Secondary metrics: **number_acc** (numeric only), **unit_acc** (unit only)
- Two evaluators: ``interval_matching`` (single valid range) and ``multi_interval_matching`` (e.g. clock AM/PM)
- Model output is expected in the format ``Answer: <value> <unit>`` on the last line
- ``image_type`` is recorded in each sample's metadata; per-type results are visible in the
  ``subset_key`` column of review files but are not separately selectable via ``subset_list``
- [Paper](https://arxiv.org/abs/2510.26865) | [GitHub](https://github.com/flageval-baai/MeasureBench)
"""

ANSWER_FORMAT_SUFFIX = (
    '\nProvide your final answer on the last line in the format: Answer: <value> <unit>. '
    'For example: Answer: 42.5 A'
)


@register_benchmark(
    BenchmarkMeta(
        name='measure_bench',
        pretty_name='MeasureBench',
        dataset_id='evalscope/MeasureBench',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2510.26865',
        subset_list=['real_world', 'synthetic_test'],
        metric_list=['acc', 'number_acc', 'unit_acc'],
        eval_split='real_world',
    )
)
class MeasureBenchAdapter(VisionLanguageAdapter):
    """Data adapter for evalscope/MeasureBench.

    Supports two dataset splits (real_world and synthetic_test), each treated as a
    separate top-level subset via ``split_as_subset=True``.  Images are PIL objects
    stored in the parquet files; they are converted to base64 JPEG for API inference.
    Scoring is deterministic: a numeric/time interval comparison with optional unit check.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Treat dataset splits as top-level subsets
        self.split_as_subset = True

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Optional[Sample]:
        """Convert a raw MeasureBench record to a multimodal Sample.

        The ``image`` field may be a PIL Image (from parquet) or a bytes dict.
        ``evaluator_kwargs`` is a JSON string containing the scoring parameters.
        """
        image_field = record.get('image')
        if image_field is None:
            logger.warning(f'Record {record.get("question_id")} has no image; skipping.')
            return None

        # evalscope RemoteDataLoader sets decode=False on Image columns, so the
        # field arrives as a bytes dict {'bytes': b'...', 'path': ...}.
        # Direct load_dataset_from_hub (without RemoteDataLoader) returns PIL.
        if isinstance(image_field, dict) and image_field.get('bytes'):
            image_b64 = self._image_bytes_to_base64(image_field['bytes'], default_format='jpeg')
        elif hasattr(image_field, 'save'):  # PIL Image fallback
            from evalscope.utils.io_utils import PIL_to_base64
            image_b64 = PIL_to_base64(image_field.convert('RGB'), format='JPEG', add_header=True)
        else:
            logger.warning(
                f'Record {record.get("question_id")} has unsupported image type {type(image_field)}; skipping.'
            )
            return None

        question: str = record.get('question', '')
        prompt_text = f'{question}{ANSWER_FORMAT_SUFFIX}'

        content_list: List[Content] = [
            ContentImage(image=image_b64),
            ContentText(text=prompt_text),
        ]

        # Evaluator config — keep as raw JSON string; parse only at score time
        evaluator: str = record.get('evaluator', 'interval_matching')
        evaluator_kwargs_raw: str = record.get('evaluator_kwargs', '{}')

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target='',  # Ground truth is encoded in evaluator_kwargs
            subset_key=record.get('image_type', ''),
            metadata={
                'question_id': record.get('question_id', ''),
                'image_type': record.get('image_type', ''),
                'design': record.get('design', ''),
                'evaluator': evaluator,
                'evaluator_kwargs': evaluator_kwargs_raw,
            },
        )

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract the instrument reading from the model's raw output.

        Looks for ``Answer:``/``Answer`` markers and ``\\boxed{...}`` patterns.
        """
        from .utils import extract_answer_text, normalize_string

        return normalize_string(extract_answer_text(prediction))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Evaluate the instrument reading using interval matching.

        Returns a Score with:
        - ``acc`` (1.0 / 0.0)  — number AND unit both correct (``all_correct``)
        - ``number_acc``        — numeric value within the tolerance interval
        - ``unit_acc``          — recognised unit present in the response (None if no unit required)
        """
        from .utils import interval_matching, multi_interval_matching

        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)

        metadata = task_state.metadata or {}
        evaluator: str = metadata.get('evaluator', 'interval_matching')
        evaluator_kwargs_raw: str = metadata.get('evaluator_kwargs', '{}')

        try:
            evaluator_kwargs: Dict[str, Any] = json.loads(evaluator_kwargs_raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f'Failed to parse evaluator_kwargs: {evaluator_kwargs_raw!r}')
            score.value = {'acc': 0.0, 'number_acc': 0.0, 'unit_acc': 0.0}
            score.main_score_name = 'acc'
            return score

        try:
            if evaluator == 'interval_matching':
                result = interval_matching(
                    answer_text=filtered_prediction,
                    interval=evaluator_kwargs['interval'],
                    units=evaluator_kwargs.get('units', []),
                )
            elif evaluator == 'multi_interval_matching':
                result = multi_interval_matching(
                    answer_text=filtered_prediction,
                    intervals=evaluator_kwargs['intervals'],
                    units=evaluator_kwargs.get('units', []),
                )
            else:
                logger.warning(f'Unknown evaluator: {evaluator!r}; assigning zero score.')
                result = {'all_correct': 0, 'number_correct': 0, 'number_error_rate': None, 'unit_correct': 0}
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f'Scoring error for question {metadata.get("question_id")}: {exc}')
            result = {'all_correct': 0, 'number_correct': 0, 'number_error_rate': None, 'unit_correct': 0}

        score_value: Dict[str, float] = {
            'acc': float(result.get('all_correct', 0)),
            'number_acc': float(result.get('number_correct', 0)),
        }
        unit_correct = result.get('unit_correct')
        if unit_correct is not None:
            score_value['unit_acc'] = float(unit_correct)

        score.value = score_value
        score.main_score_name = 'acc'
        return score
