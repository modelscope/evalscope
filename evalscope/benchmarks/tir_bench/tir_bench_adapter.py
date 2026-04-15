# flake8: noqa: E501
import os
import zipfile
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = """
## Overview

TIR-Bench (Thinking-with-Images Reasoning Benchmark) is a comprehensive multimodal benchmark
that evaluates agentic visual reasoning capabilities of vision-language models. It covers
diverse task categories requiring spatial, compositional, and multi-step visual reasoning.

## Task Description

- **Task Type**: Multi-task Visual Reasoning (MCQ, OCR, word search, spot difference, jigsaw, etc.)
- **Input**: One or two images + question (most tasks use multiple-choice format)
- **Output**: Answer letter (MCQ) or numeric/text response depending on task type
- **Domains**: instrument, color, refcoco, rotation_game, math, word_search, visual_search, ocr, symbolic, spot_difference, contrast, jigsaw, maze

## Key Features

- 1,215 test samples across 13 diverse visual reasoning task categories
- Covers single-image and dual-image reasoning scenarios
- Answers span letter choices (A-J), integers, floats, and text
- Task-specific scoring with LLM-as-judge fallback for robust evaluation

## Evaluation Notes

- Default evaluation uses the **test** split (1,215 samples)
- Primary metric: **Accuracy** (acc)
- Images are downloaded as `data.zip` from ModelScope and extracted automatically
- Rule-based scoring: OCR (substring match), jigsaw (grid IoU), spot_difference (set IoU),
  word_search (numeric match), all other tasks (MCQ / numeric judge)
- **Recommended**: set `judge_strategy=JudgeStrategy.LLM_RECALL` and provide `judge_model_args`
  to activate LLM-as-judge as a recall mechanism — the judge is called only when rule-based
  scoring gives 0, providing more accurate evaluation without unnecessary API overhead
- [Paper](https://arxiv.org/abs/2511.01833) | [GitHub](https://github.com/agents-x-project/TIR-Bench)
"""


@register_benchmark(
    BenchmarkMeta(
        name='tir_bench',
        pretty_name='TIR-Bench',
        dataset_id='evalscope/TIR-Bench',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2511.01833',
        subset_list=[
            'instrument',
            'color',
            'refcoco',
            'rotation_game',
            'math',
            'word_search',
            'visual_search',
            'ocr',
            'symbolic',
            'spot_difference',
            'contrast',
            'jigsaw',
            'maze',
        ],
        metric_list=['acc'],
        eval_split='test',
    )
)
class TIRBenchAdapter(VisionLanguageAdapter):
    """Data adapter for evalscope/TIR-Bench.

    Handles:
    - Downloading and extracting the image archive (data.zip) via
      ``modelscope.dataset_snapshot_download``.
    - Converting raw dataset records into multimodal :class:`Sample` objects
      with base64-encoded images.
    - Task-aware answer extraction and scoring.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set during load(); used by record_to_sample to resolve image paths.
        self.image_root: Optional[str] = None
        self.reformat_subset = True

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load(self):
        """Download data.zip (if needed), extract images, then load the dataset."""
        dataset_name_or_path = self.dataset_id

        if os.path.exists(dataset_name_or_path):
            dataset_path = dataset_name_or_path
            logger.info(f'Loading TIR-Bench from local path: {dataset_path}')
        else:
            from modelscope import dataset_snapshot_download
            logger.info(f'Downloading TIR-Bench dataset from ModelScope: {dataset_name_or_path}')
            dataset_path = dataset_snapshot_download(
                dataset_name_or_path,
                allow_file_pattern=['data.zip'],
            )

        # Save for use in record_to_sample
        self.image_root = dataset_path

        # Extract data.zip if the extracted directory does not yet exist
        extract_dir = os.path.join(dataset_path, 'data')
        zip_path = os.path.join(dataset_path, 'data.zip')
        if not os.path.exists(extract_dir):
            if os.path.exists(zip_path):
                logger.info(f'Extracting {zip_path} to {dataset_path} ...')
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(dataset_path)
                logger.info('Extraction complete.')
            else:
                logger.warning(
                    f'data.zip not found at {zip_path}. '
                    'Image loading may fail if images are not already extracted.'
                )

        return self.load_from_remote()

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a raw TIR-Bench record into a multimodal Sample.

        Images (image_1 and optionally image_2) are loaded from the extracted
        data directory and encoded as base64 data URIs. The model prompt is
        already fully formatted in the dataset (includes choices for MCQ tasks).
        """
        content_list: List[Content] = []

        for img_key in ('image_1', 'image_2'):
            img_path = record.get(img_key)
            if not img_path:
                continue
            full_path = os.path.join(self.image_root, img_path)
            if not os.path.exists(full_path):
                logger.warning(f'Image not found: {full_path}')
                continue
            ext = os.path.splitext(img_path)[1].lower().lstrip('.')
            if not ext:
                ext = 'jpg'
            with open(full_path, 'rb') as fh:
                image_b64 = bytes_to_base64(fh.read(), format=ext, add_header=True)
            content_list.append(ContentImage(image=image_b64))

        content_list.append(ContentText(text=record['prompt']))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=str(record['answer']),
            subset_key=record.get('task', ''),
            metadata={
                'task': record.get('task', ''),
                'meta_data': record.get('meta_data') or {},
                'id': record.get('id'),
            },
        )

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract the answer from the model's raw prediction.

        Dispatches based on :func:`classify_string` applied to the reference
        answer, so the right extraction strategy is chosen per answer type:

        - **ocr**: full prediction is returned unchanged; scoring uses substring
          match on the raw response.
        - **type 1** (alphabetic / MCQ letter): :func:`extract_mcq_answer` is
          called.  It handles ``\\boxed{X}``, ``Answer: X`` markers, and
          ``(X)`` parenthesised letters (always taking the *last* occurrence to
          avoid being confused by option listings in the reasoning text).
        - **type 2 / 3 / 4** (integer / float / composite such as ``[2, 3]``):
          the raw prediction is returned so that the task-specific scoring
          helpers in :meth:`match_score` (``judge_int``, ``judge_float``,
          ``extract_two_numbers``, ``extract_consecutive_integers``, etc.) can
          apply their own targeted extraction.
        """
        task = (task_state.metadata or {}).get('task', '')
        if task == 'ocr':
            return prediction

        from .utils import extract_answer_with_classify
        return extract_answer_with_classify(prediction, task_state.target)

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
        """Task-aware scoring faithful to the original TIR-Bench evaluation.

        Dispatches to task-specific scoring logic:
        - ``ocr``            : substring match on the raw model response
        - ``word_search``    : integer comparison or two-number tuple match
        - ``spot_difference``: set IoU over extracted integer lists
        - ``jigsaw``         : position-wise accuracy of n² grid permutations
        - everything else    : MCQ letter / integer / float judge based on
                               ``classify_string(reference)``
        """
        from .utils import (
            classify_string,
            compare,
            extract_consecutive_integers,
            extract_consecutive_n_squared,
            extract_two_numbers,
            judge_choice,
            judge_float,
            judge_int,
            list_iou,
        )

        metadata = task_state.metadata or {}
        task = metadata.get('task', '')
        meta_data = metadata.get('meta_data') or {}
        prompt_text = task_state.input_text or ''
        answer = reference

        correctness = 0.0

        try:
            if task == 'ocr':
                # Special-case: check if answer appears as substring in raw response
                if answer in original_prediction:
                    correctness = 1.0

            elif task == 'word_search':
                if classify_string(answer) == 2:
                    correctness = judge_int(filtered_prediction, answer)
                else:
                    try:
                        a1, a2 = extract_two_numbers(answer)
                        r1, r2 = extract_two_numbers(filtered_prediction)
                        correctness = 1.0 if a1 == r1 and a2 == r2 else 0.0
                    except (TypeError, ValueError):
                        pass

            elif task == 'spot_difference':
                if classify_string(answer) == 2:
                    correctness = judge_int(filtered_prediction, answer)
                else:
                    try:
                        list_answer = extract_consecutive_integers(answer)
                        list_response = extract_consecutive_integers(filtered_prediction)
                        correctness = list_iou(list_response, list_answer)
                    except Exception:
                        pass

            elif task == 'jigsaw':
                try:
                    difficulty = int(meta_data.get('difficulty', 2))
                    a_re = extract_consecutive_n_squared(answer, difficulty)
                    m_re = extract_consecutive_n_squared(filtered_prediction, difficulty)
                    correctness = compare(a_re, m_re)
                except Exception:
                    pass

            else:
                # Generic dispatch based on answer type
                string_type = classify_string(answer)
                if string_type == 1:
                    correctness = judge_choice(filtered_prediction, answer, prompt_text)
                elif string_type == 2:
                    correctness = judge_int(filtered_prediction, answer)
                elif string_type == 3:
                    correctness = judge_float(filtered_prediction, answer)

        except Exception as exc:
            logger.warning(f'[TIR-Bench] Scoring error for task={task!r}: {exc}')

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        score.value = {'acc': correctness}
        score.main_score_name = 'acc'
        return score
