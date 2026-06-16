import json
import re
import string
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """Answer the question according to the image using a short phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes)."""

_ANSWER_PATTERN = re.compile(r'ANSWER:\s*(.*)', flags=re.IGNORECASE)
_ARTICLES = {'a', 'an', 'the'}
_NUMBER_MAP = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
}


def _normalize_vqa_answer(answer: str) -> str:
    text = answer.lower().strip().replace('\n', ' ').replace('\t', ' ')
    punctuation = string.punctuation.replace(':', '')
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = ''.join(' ' if c in punctuation else c for c in text)
    words = []
    for word in text.split():
        mapped = _NUMBER_MAP.get(word, word)
        if mapped not in _ARTICLES:
            words.append(mapped)
    return ' '.join(words)


def _vqa_soft_accuracy(prediction: str, answers: List[str]) -> float:
    norm_pred = _normalize_vqa_answer(prediction)
    if not norm_pred:
        return 0.0
    count = sum(_normalize_vqa_answer(a) == norm_pred for a in answers)
    return min(1.0, count / 3.0)


def _vqa_exact_match(prediction: str, answers: List[str]) -> float:
    norm_pred = _normalize_vqa_answer(prediction)
    return float(norm_pred in {_normalize_vqa_answer(a) for a in answers})


def _ensure_answer_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        results = []
        for item in value:
            if isinstance(item, str):
                results.append(item)
            elif isinstance(item, dict):
                results.append(str(item.get('answer') or item.get('text') or ''))
        return [r for r in results if r.strip()]
    return [str(value)]


@register_benchmark(
    BenchmarkMeta(
        name='vqav2',
        pretty_name='VQAv2',
        description="""
## Overview

VQAv2 is the balanced Visual Question Answering benchmark built on COCO images. It evaluates whether
multimodal models can answer open-ended natural-language questions grounded in image content.

## Task Description

- **Task Type**: Open-ended visual question answering
- **Input**: Image + natural-language question
- **Output**: Short answer phrase
- **Domains**: General image understanding, object recognition, counting, attributes, relations

## Evaluation Notes

- Default data source: `lmms-lab/VQAv2` on ModelScope, `validation` split
- Primary metric: **VQAv2 soft accuracy** over human annotator answers
- Also reports normalized exact match against the available answer set
- The adapter accepts common answer formats: list of strings, list of answer dicts, or `multiple_choice_answer`
""",
        tags=[Tags.MULTI_MODAL, Tags.QA],
        dataset_id='lmms-lab/VQAv2',
        paper_url='https://arxiv.org/abs/1612.00837',
        subset_list=['default'],
        metric_list=['vqa_score', 'exact_match'],
        eval_split='validation',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class VQAv2Adapter(VisionLanguageAdapter):
    """Adapter for VQAv2 open-ended visual question answering."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_aggregation_name = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = str(record.get('question') or '').strip()
        answers = _ensure_answer_list(record.get('answers'))
        if not answers:
            answers = _ensure_answer_list(record.get('multiple_choice_answer') or record.get('answer'))

        content_list: List[Content] = [ContentText(text=self.prompt_template.format(question=question))]
        image = record.get('image')
        if image:
            if isinstance(image, dict) and image.get('bytes'):
                image_b64 = self._image_bytes_to_base64(image['bytes'], default_format='jpeg')
            elif isinstance(image, bytes):
                image_b64 = self._image_bytes_to_base64(image, default_format='jpeg')
            else:
                image_b64 = str(image)
            content_list.append(ContentImage(image=image_b64))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(answers, ensure_ascii=False),
            metadata={
                'question': question,
                'answers': answers,
                'multiple_choice_answer': record.get('multiple_choice_answer'),
                'question_id': record.get('question_id'),
                'question_type': record.get('question_type'),
                'answer_type': record.get('answer_type'),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        matches = _ANSWER_PATTERN.findall(prediction or '')
        if matches:
            return matches[-1].strip().strip('"\'')
        return (prediction or '').strip().strip('"\'')

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        answers = task_state.metadata.get('answers') or []
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value = {
            'vqa_score': _vqa_soft_accuracy(filtered_prediction, answers),
            'exact_match': _vqa_exact_match(filtered_prediction, answers),
        }
        score.main_score_name = 'vqa_score'
        return score
