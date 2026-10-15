import json
from pathlib import Path
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()
EVALUATOR_URL = 'https://huggingface.co/spaces/paulgavrikov/visualoverload-submit'


@register_benchmark(
    BenchmarkMeta(
        name='visualoverload',
        pretty_name='VisualOverload',
        dataset_id='evalscope/visualoverload',
        paper_url='https://arxiv.org/abs/2509.25339',
        tags=[Tags.MULTI_MODAL, Tags.QA, Tags.REASONING],
        subset_list=['default'],
        eval_split='test',
        few_shot_num=0,
        few_shot_mode='disabled',
        metric_list=['response_present'],
        evaluation_version='v1.0',
        description="""
## Overview

VisualOverload (CVPR 2026) probes visual understanding in dense, high-resolution scenes.
Ground-truth answers are privately held; official grading uses the benchmark's evaluation server.

## Task Description

- **Task Type**: Visual question answering
- **Input**: A high-resolution painting and the official `default_prompt`
- **Output**: An option letter, a count, or an OCR answer, depending on the question
- **Domain**: Activities, attributes, counting, OCR, reasoning, and scene understanding

## Key Features

- 2,720 questions from `evalscope/visualoverload` on ModelScope, mirrored from the official
  `paulgavrikov/visualoverload` on Hugging Face
- Combines multiple-choice and free-form questions across easy, medium, and hard difficulties
- Uses the official prompt unchanged and preserves original image bytes by default
- Exports raw responses in the official `question_id`/`response` submission format

## Evaluation Notes

- The dataset is loaded from the ModelScope mirror's public `test` split with zero-shot evaluation.
- `response_present` is a diagnostic for non-empty responses, not answer accuracy.
- Official scores are available only from the evaluation server; no local quality metric is selected.
- Upload `visualoverload_submission.json` from the model's reports directory to
  [the official evaluator](https://huggingface.co/spaces/paulgavrikov/visualoverload-submit).
- Hugging Face sign-in is required for official evaluation; EvalScope does not submit automatically.
- Use one generation per question (`repeats=1`). Duplicate question IDs are rejected during export.
- Limited runs produce partial submissions for testing, not full benchmark scores.
- Model processors may resize or tile images; report those settings. `max_image_bytes` can explicitly
  enable compression, which changes the visual input.
""",
    )
)
class VisualOverloadAdapter(VisionLanguageAdapter):
    """Generate VisualOverload predictions and export them for private server grading."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Use original image bytes and the official prompt without a public target."""
        image = record['image']
        if image.get('bytes') is not None:
            image_value = self._image_bytes_to_base64(image['bytes'], default_format='jpeg', guess_mimetype=True)
        else:
            image_value = self._image_bytes_to_base64(
                Path(image['path']).read_bytes(), default_format='jpeg', guess_mimetype=True
            )
        return Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentImage(image=image_value),
                        ContentText(text=record['default_prompt']),
                    ]
                )
            ],
            target='',
            metadata={key: record[key] for key in ('question_id', 'question_type', 'category', 'difficulty')},
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Track response availability only; private answers cannot be graded locally."""
        return Score(
            prediction=original_prediction,
            extracted_prediction=original_prediction,
            value={'response_present': float(bool(original_prediction.strip()))},
            main_score_name='response_present',
            metadata={
                'response_present': bool(original_prediction.strip()),
                'official_score': None,
                'evaluator_url': EVALUATOR_URL,
            },
        )

    def _on_generate_report_end(self, report: Any, output_dir: str, **kwargs: Any) -> None:
        report_dir = Path(output_dir)
        review_dir = report_dir.parent.parent / 'reviews' / report_dir.name
        records: List[Dict[str, str]] = []
        for subset in self.subset_list:
            review_file = review_dir / f'{self.name}_{subset}.jsonl'
            for item in jsonl_to_list(str(review_file)):
                sample_score = item['sample_score']
                question_id = sample_score['sample_metadata']['question_id']
                if question_id is None:
                    raise ValueError('VisualOverload submissions require non-null question_id values.')
                records.append(
                    {
                        'question_id': str(question_id),
                        'response': sample_score['score'].get('prediction') or '',
                    }
                )
        question_ids = [record['question_id'] for record in records]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError('VisualOverload submissions require unique question_id values; use repeats=1.')
        submission_file = report_dir / 'visualoverload_submission.json'
        report_dir.mkdir(parents=True, exist_ok=True)
        with submission_file.open('w', encoding='utf-8') as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2)
        logger.info(f'VisualOverload answers are private. Submit {submission_file} to {EVALUATOR_URL}')
