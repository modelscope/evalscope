import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentText
from evalscope.api.metric import Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.utils.rouge import compute_rouge_score_one_sample_zh
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    'Retrieval',
    'Reasoning',
    'Memory',
]
PROMPTS = {
    'Retrieval': (
        'Answer a question based on the above book snippet.'
        ' Some special magic numbers are hidden within the following text.'
        ' Make sure to memorize it. I will quiz you about the numbers afterwards. Question: '
    ),
    'Reasoning': (
        'Answer a question based on the above book snippet.'
        ' Your answer should be short and based on either explicitly stated facts or strong, logical inferences.'
        ' Return only the final answer with no additional explanation or reasoning. Question: '
    ),
    'Memory': (
        'Based on the above context, write an answer in the form of a short phrase for the following question.'
        ' Answer with exact words from the context whenever possible. Question: '
    ),
}
DESCRIPTION = """
## Overview

VTCBench (Vision-Text Compression Benchmark) evaluates VLMs' ability to compress visual text.

## Task Description

- **Task Type**: Visual question answering with dual evaluation modes
- **Input**: Either (VTC) image(s) + problem text, or (Text) text context + problem text
- **Output**: Short free-form answer
- **Domain**: General visual comprehension, text-rich image understanding

## Key Features

- Dual evaluation modes: image-based (VTC) and text-based (Text)
- Mode VTC tests the model's visual understanding by feeding images directly
- Mode Text tests the model's text-based reasoning using the image's textual context
- The Gap highlights the model's ability to leverage visual information versus textual context for question answering

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Long context benchmark requires longer interval, set `retry_interval` higher to avoid timeout
- Use `--dataset-args '{"vtcbench": {"extra_params":{"eval_mode":"text"}}}'` to switch modes, default is 'vtc'
- Metrics:
  - **containsAll**/**ROUGE-1-R** for Retrieval and Reasoning subsets
  - **ROUGE-L-R**/**LLM-Judge** for Memory subset
- If you encounter casting offset overflow issues, set `DATASET_TF_BATCH_SIZE=1`
"""


def containsAny(pred: str, answers: List[str]) -> float:
    return any(ans.lower() in pred.lower() for ans in answers)


def containsAll(pred: str, answers: List[str]) -> float:
    hit = sum(ans.lower() in pred.lower() for ans in answers)
    return hit / len(answers)


@register_benchmark(
    BenchmarkMeta(
        name='vtcbench',
        pretty_name='VTCBench',
        description=DESCRIPTION,
        tags=[Tags.MULTI_MODAL, Tags.QA, Tags.LONG_CONTEXT, Tags.RETRIEVAL, Tags.REASONING],
        dataset_id='MLLM-CL/VTCBench',
        subset_list=SUBSET_LIST,
        metric_list=['Rouge'],
        primary_metric=MetricSelector(
            name='rouge', aggregation='mean', dimensions={
                'variant': 'l',
                'statistic': 'recall',
            }
        ),
        eval_split='test',
        prompt_template=None,
        extra_params={
            'eval_mode': {
                'type': 'str',
                'description': 'Evaluation mode: vtc (images+problem) or text (text+problem).',
                'value': 'vtc',
                'choices': ['vtc', 'text'],
            },
        },
    )
)
class VTCBenchAdapter(VisionLanguageAdapter):
    """Adapter for VTCBench dual-mode visual question answering."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eval_mode: str = self.extra_params.get('eval_mode', 'vtc')

        check_import(
            module_name=['bs4'],
            extra='docs',
            raise_error=True,
            feature_name='VTCBench',
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        from bs4 import BeautifulSoup

        # the qa for the context
        problem: str = record['problem']
        answers: list[str] = record['answers']

        eval_mode = self.eval_mode

        match eval_mode:
            case 'vtc':  # VTC = images + problem
                image_map = self._extract_media(record, media_type='image')
                content_list = self._parse_text_with_media(
                    text=''.join(f'<image {i}>' for i in range(1,
                                                               len(image_map) + 1)),
                    image_map=image_map,
                )
                assert len(content_list) == len(image_map)
                content_list = [ContentText(text=PROMPTS[self.current_subset_name])
                                ] + content_list + [ContentText(text=problem)]
            case 'text':  # text = _context + problem
                if self.current_subset_name != 'Memory':
                    _context = record['_context']
                else:
                    _context = re.sub(r'<image \d+>', '', record['_context'])
                    soup = BeautifulSoup(_context, 'html.parser')
                    lines = []
                    for span in soup.select('span'):
                        speaker = span.get('data-speaker')
                        text = span.get_text(' ', strip=True)
                        if speaker:
                            lines.append(f'{speaker}: {text}')
                        else:
                            lines.append(text)
                    _context = '\n'.join(lines)
                content_list = [
                    ContentText(text=PROMPTS[self.current_subset_name]),
                    ContentText(text=_context),
                    ContentText(text=problem),
                ]
            case _:
                raise ValueError(f'Unsupported eval_mode: {eval_mode}. Use "vtc" or "text".')

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=' '.join(answers),
            metadata={
                'problem': problem,
                'answers': answers,
                'subset': self.current_subset_name,
                'eval_mode': eval_mode,
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            main_score_name='Rouge-L-R' if task_state.metadata.get('subset') == 'Memory' else 'containsAll',
        )

        for metric in self.metric_list:
            try:
                score.value['containsAny'] = containsAny(original_prediction, task_state.metadata.get('answers', []))
                score.value['containsAll'] = containsAll(original_prediction, task_state.metadata.get('answers', []))
                score.value.update(compute_rouge_score_one_sample_zh([original_prediction], [reference]))
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                return None

        return score
