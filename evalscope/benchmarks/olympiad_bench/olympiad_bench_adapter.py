from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.messages.content import Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    'OE_MM_maths_en_COMP',
    'OE_MM_maths_zh_CEE',
    'OE_MM_maths_zh_COMP',
    'OE_MM_physics_en_COMP',
    'OE_MM_physics_zh_CEE',
    'OE_TO_maths_en_COMP',
    'OE_TO_maths_zh_CEE',
    'OE_TO_maths_zh_COMP',
    'OE_TO_physics_en_COMP',
    'OE_TO_physics_zh_CEE',
    'TP_MM_maths_en_COMP',
    'TP_MM_maths_zh_CEE',
    'TP_MM_maths_zh_COMP',
    'TP_MM_physics_en_COMP',
    'TP_TO_maths_en_COMP',
    'TP_TO_maths_zh_CEE',
    'TP_TO_maths_zh_COMP',
    'TP_TO_physics_en_COMP',
]


@register_benchmark(
    BenchmarkMeta(
        name='olympiad_bench',
        pretty_name='OlympiadBench',
        tags=[Tags.MATH, Tags.REASONING],
        description="""
## Overview

OlympiadBench is an Olympiad-level bilingual multimodal scientific benchmark featuring 8,476 problems from mathematics and physics competitions, including the Chinese college entrance exam (CEE). It provides rigorous evaluation of advanced scientific reasoning.

## Task Description

- **Task Type**: Olympiad-Level Math/Physics Problem Solving
- **Input**: Problem text with optional images (up to 9)
- **Output**: Mathematical answer or proof
- **Domains**: Mathematics, Physics (bilingual: English and Chinese)

## Key Features

- 8,476 Olympiad-level problems
- Bilingual support (English and Chinese)
- Covers both Mathematics and Physics
- Subset naming convention:
  - `OE`: Open-Ended problems
  - `TP`: Theorem Proving problems
  - `MM`: Multimodal (with images)
  - `TO`: Text-Only
  - `CEE`: Chinese Entrance Exam
  - `COMP`: Comprehensive competition problems

## Evaluation Notes

- Default evaluation uses the **train** split
- Primary metric: **Accuracy** with mathematical judging
- Answers should be in \\boxed{} format
- **Note**: `TP` (Theorem Proving) subsets cannot be auto-evaluated currently
- Supports numerical precision/error thresholds for approximate answers
""",
        dataset_id='AI-ModelScope/OlympiadBench',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='train',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
        evaluation_version='v1.2',
    )
)
class OlympiadBenchAdapter(VisionLanguageAdapter):
    MAX_IMAGES: int = 9

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Generate prompt for a single item."""
        from .utils import OlympiadBenchPrompter

        question = record.get('question', '')
        language = record.get('language', 'English')
        subject = record.get('subject', 'Math')
        question_type = record.get('question_type', '')
        answer_type = record.get('answer_type', '')
        is_multiple_answer = record.get('is_multiple_answer', False)
        unit = record.get('unit', '')
        # Generate prompt
        prompt = OlympiadBenchPrompter().make_prompt(
            problem=question,
            language=language,
            subject=subject,
            question_type=question_type,
            answer_type=answer_type,
            is_multiple_answer=is_multiple_answer,
            unit=unit,
        )
        # Construct content list
        content_list: List[Content] = []
        for index, image in self._extract_media(record, 'image').items():
            content_list.append(self._content_image_from_value(image))
            prompt = prompt.replace(f'<image_{index}>', f'[image_{index}]')
        # Add text content
        content_list.insert(0, ContentText(text=prompt))

        final_answer = record.get('final_answer', [])
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=','.join(final_answer) if final_answer else '',
            metadata={
                'id': record.get('id', ''),
                'subfield': record.get('subfield', ''),
                'context': record.get('context', ''),
                'solution': record.get('solution', []),
                'final_answer': record.get('final_answer', []),
                'is_multiple_answer': is_multiple_answer,
                'unit': unit,
                'answer_type': answer_type,
                'question_type': question_type,
                'language': language,
                'subject': subject,
                'error': record.get('error', None),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        from evalscope.metrics.math.parser import extract_answer

        return extract_answer(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from evalscope.constants import ScoreStatus
        from evalscope.metrics.math.contracts import MathEvaluationError
        from evalscope.metrics.math.parser import compare_answers

        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        metadata = task_state.metadata
        try:
            error = metadata.get('error')
            tolerances = (
                [float(p) if p.strip() else 1e-8 for p in str(error).split(',')] if error is not None else [1e-8]
            )
            answer_type = metadata.get('answer_type', '')
            if 'Tuple' in answer_type:
                tolerances = [1e-8]
            # Preserve the benchmark's official reference group; subsequent entries
            # are alternative derivations, not additional required answers.
            official_answers = metadata.get('final_answer') or []
            reference = official_answers[0] if official_answers else reference
            result = compare_answers(
                filtered_prediction,
                reference,
                absolute_tolerance=tolerances,
                multiple_answers=metadata.get('is_multiple_answer', False) or ',' in answer_type,
            )
            score.value = {'acc': float(result.matched)}
            score.metadata['math_reason'] = result.reason
        except (MathEvaluationError, ValueError) as exc:
            score.status = ScoreStatus.EXCLUDED
            score.metadata['metric_unavailable'] = True
            score.metadata['acc'] = f'error: {exc}'
        return score
