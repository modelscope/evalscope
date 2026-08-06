# flake8: noqa: E501
import base64
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser, Content
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = """
## Overview

PerceptionBench is a benchmark from Moonshot AI that evaluates the atomic visual perception
capabilities of multimodal large language models. It is built bottom-up: the earliest failure
points of frontier MLLMs on 42 existing benchmarks were diagnosed to derive an error taxonomy
whose perception branch defines ten atomic perceptual capabilities. Each question isolates a
single capability, so difficulty stems from perception rather than reasoning or knowledge.

## Task Description

- **Task Type**: Visual Perception (open-ended question answering)
- **Input**: One or more images interleaved with a question
- **Output**: Free-form short answer with a uniquely determined reference
- **Domain**: Atomic visual perception across ten capabilities

## Key Features

- 3,000 verified questions covering ten atomic perceptual capabilities
- 1,800 questions (60%) are atomic sub-questions decomposed from attributed failures on source
  benchmarks; 1,200 (40%) are newly authored on supplemented images
- Subsets follow the ten `error_category` labels: visual relation, counting, attribute,
  depth & 3D perception, localization, comparison, fine-grained recognition, contextual
  integration, OCR, and perception-related hallucination
- Multi-image questions are supported: images are interleaved into the question via
  `<|image_N|>` placeholders
- Samples carrying a `hint` (coordinate convention or image dimensions) pass it as a system
  message, matching the official message builder

## Evaluation Notes

- Default evaluation uses the **train** split (3,000 samples, single split dataset)
- Primary metric: **Accuracy**, reported overall and per capability
- Scoring follows the official protocol: an LLM judge grades the free-form answer against the
  reference with the teacher-grading prompt and returns a strict 0/1 verdict per item
  (`[reason]` / `[judge] True|False`); the paper uses GPT-oss-120B, whose agreement with human
  judgment is 99.7% on a 300-sample audit
- Empty or failed generations are scored 0 without invoking the judge
- Requires `judge_model_args` configuration for the LLM judge
- The dataset embeds images as base64 data URIs (~1.6 GB download on first use)
"""

SUBSET_LIST = [
    'visual_relation_error',
    'visual_counting_error',
    'visual_attribute_error',
    'depth_3d_perception_error',
    'visual_localization_error',
    'visual_comparison_error',
    'fine_grained_recognition_error',
    'context_integration_error',
    'ocr_error',
    'hallucination',
]


@register_benchmark(
    BenchmarkMeta(
        name='perception_bench',
        pretty_name='PerceptionBench',
        dataset_id='moonshotai/PerceptionBench',
        tags=[Tags.MULTI_MODAL, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2607.24957',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='train',
    )
)
class PerceptionBenchAdapter(VisionLanguageAdapter):
    """Data adapter for moonshotai/PerceptionBench.

    Interleaves the question with its images following the official `<|image_N|>`
    placeholder convention and scores free-form answers with the official LLM judge.
    """
    llm_judge_default = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Optional[Sample]:
        """Convert a PerceptionBench record to a Sample."""
        problem: str = record.get('problem', '')
        images: List[str] = record.get('image') or []

        # Official placeholders are <|image_N|>; normalize them to the framework syntax.
        text = problem
        image_map = {}
        for idx, image in enumerate(images, start=1):
            text = text.replace(f'<|image_{idx}|>', f'<image_{idx}>')
            image_map[idx] = self._data_uri_to_base64(image)

        content_list: List[Content] = self._parse_text_with_images(text=text, image_map=image_map)
        if not content_list:
            logger.warning(f'Record {record.get("index")} has no usable content, skipping.')
            return None

        messages = []
        hint: str = record.get('hint') or ''
        if hint.strip():
            messages.append(ChatMessageSystem(content=hint))
        messages.append(ChatMessageUser(content=content_list))

        return Sample(
            input=messages,
            target=str(record.get('answer', '')),
            subset_key=record.get('error_category', ''),
            metadata={
                'index': record.get('index'),
                'problem': problem,
                'error_category': record.get('error_category', ''),
                'source_bmk': record.get('source_bmk', ''),
                'source_idx': record.get('source_idx'),
            },
        )

    def _data_uri_to_base64(self, image: str) -> str:
        """Re-encode a base64 data URI so the shared image size limit applies."""
        header, _, payload = image.partition(',')
        if not payload:
            # Not a data URI (e.g. a plain URL): pass through unchanged.
            return image
        image_format = header.split('/')[-1].split(';')[0] or 'jpeg'
        return self._image_bytes_to_base64(base64.b64decode(payload), default_format=image_format)

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score a prediction with the official PerceptionBench teacher-grading judge."""
        from .utils import build_judge_prompt, parse_judge_verdict

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        if not (original_prediction or '').strip():
            # Mirrors the official evaluator: unanswered items score 0 without a judge call.
            score.value = {'acc': 0.0}
            score.explanation = 'failed to obtain answer'
            return score

        metadata = task_state.metadata or {}
        prompt = build_judge_prompt(
            question=metadata.get('problem', task_state.input_text),
            prediction=original_prediction,
            reference=reference,
        )
        judge_response = self.llm_judge.judge(prompt)
        judge_score, judge_reason = parse_judge_verdict(judge_response)

        score.value = {'acc': judge_score}
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
            'judge_reason': judge_reason,
        }
        return score
