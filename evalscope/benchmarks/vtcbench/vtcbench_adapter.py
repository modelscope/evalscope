import re
from typing import Any, Dict, List

from rouge_score import rouge_scorer

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentText
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import

SUBSET_LIST = [
    'Retrieval',
    'Reasoning',
    'Memory',
]

# These parts reproduce the official task templates around the pre-rendered context.
PROMPT_PREFIXES = {
    'Retrieval': '',
    'Reasoning': 'You will answer a question based on the following book snippet:\n\n',
    'Memory': '',
}
PROMPT_SUFFIXES = {
    'Retrieval': '\n\nQuestion:{question}',
    'Reasoning': (
        '\n\nUse the information provided in the book snippet to answer the question.'
        ' Your answer should be short and based on either explicitly stated facts or strong, logical inferences.'
        '\n\nQuestion: {question}\n\n Return only the final answer with no additional explanation or reasoning.'
    ),
    'Memory': (
        '\n\nQuestion:Based on the above context, write an answer in the form of a short phrase for the following'
        ' question. Answer with exact words from the context whenever possible.\n\n{question}'
    ),
}

DESCRIPTION = """
## Overview

VTCBench (Vision-Text Compression Benchmark) evaluates long-context understanding when text is represented as
rendered images, and compares it with a pure-text baseline.

## Task Description

- **Task Type**: Long-context question answering with image-based and text-based evaluation modes
- **Input**: Rendered context images plus a question (VTC mode), or the source text plus a question (Text mode)
- **Output**: Short free-form answer
- **Domain**: Retrieval, associative reasoning, and long-term dialogue memory

## Key Features

- Provides matched VTC and Text modes for measuring the effect of vision-text compression
- Includes Retrieval, Reasoning, and Memory subsets derived from RULER, NoLiMa, and LoCoMo
- Uses pre-rendered multi-image documents to preserve the benchmark's visual layouts
- Supports contexts spanning multiple document images

## Evaluation Notes

- Default configuration uses **0-shot** evaluation in VTC mode
- Use `--dataset-args '{"vtcbench": {"extra_params":{"eval_mode":"text"}}}'` to enable the Text baseline
- Retrieval and Reasoning use the official fractional `contains_all` score
- Memory uses the official maximum ROUGE-L F1 across reference answers
- The unified `score` metric dispatches to the official metric for each subset; its report `macro_score` is the
  unweighted mean across the three tasks
- Text mode strips HTML tags and normalizes whitespace in the same way as the official static evaluator
- Content inside `<think>...</think>` is excluded before scoring, matching the official evaluator
- Long-context requests may require a larger model timeout
- If dataset casting reports an offset overflow, set `DATASET_TF_BATCH_SIZE=1`
- [Paper](https://arxiv.org/abs/2512.15649) | [Code](https://github.com/Moenupa/VTCBench)
"""

_HTML_TAG_PATTERN = re.compile(r'<.*?>')
_WHITESPACE_PATTERN = re.compile(r'\s+')
_THINK_PATTERN = re.compile(r'<think>.*?</think>', flags=re.DOTALL)


def _remove_html_tags(text: str) -> str:
    from bs4 import BeautifulSoup

    check_import(
        module_name=['bs4'],
        extra='vtcbench',
        raise_error=True,
        feature_name='VTCBench',
    )

    if '</span>' not in text:
        duplicated_spaces = re.compile(r'\s+')
        return duplicated_spaces.sub(' ', text).strip()

    soup = BeautifulSoup(text, 'html.parser')
    lines = []
    for span in soup.select('span'):
        speaker = span.get('data-speaker')
        text = span.get_text(' ', strip=True)
        if speaker:
            lines.append(f'{speaker}: {text}')
        else:
            lines.append(text)
    _context = '\n'.join(lines)

    return _context


def _normalize_response(response: str) -> str:
    """Remove hidden reasoning and normalize case and surrounding whitespace."""
    return _THINK_PATTERN.sub('', response).strip().lower()


def _calculate_metrics(response: str, answers: List[str]) -> Dict[str, float]:
    """Calculate the official VTCBench metrics for one response."""
    if not answers:
        raise ValueError('VTCBench requires at least one reference answer.')

    normalized_answers = [str(answer).strip().lower() for answer in answers]
    normalized_response = _normalize_response(response)
    contains_any = float(any(answer in normalized_response for answer in normalized_answers))
    contains_all = sum(answer in normalized_response for answer in normalized_answers) / len(normalized_answers)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = max(
        scorer.score(target=answer, prediction=normalized_response)['rougeL'].fmeasure for answer in normalized_answers
    )
    return {
        'contains_any': contains_any,
        'contains_all': contains_all,
        'rouge_l': rouge_l,
    }


@register_benchmark(
    BenchmarkMeta(
        name='vtcbench',
        pretty_name='VTCBench',
        description=DESCRIPTION,
        tags=[Tags.MULTI_MODAL, Tags.QA, Tags.LONG_CONTEXT, Tags.RETRIEVAL, Tags.REASONING],
        dataset_id='MLLM-CL/VTCBench',
        paper_url='https://arxiv.org/abs/2512.15649',
        subset_list=SUBSET_LIST,
        metric_list=['score', 'contains_all', 'rouge_l'],
        primary_metric='normalized_score',
        eval_split='test',
        prompt_template=None,
        evaluation_version='v1.0',
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
    """Adapter for VTCBench dual-mode long-context question answering."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eval_mode: str = self.extra_params.get('eval_mode', 'vtc')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = str(record['problem'])
        answers = [str(answer) for answer in record['answers']]
        subset = self.current_subset_name
        if subset not in PROMPT_SUFFIXES:
            raise ValueError(f'Unsupported VTCBench subset: {subset}.')

        content_list = []
        prefix = PROMPT_PREFIXES[subset]
        if prefix:
            content_list.append(ContentText(text=prefix))

        if self.eval_mode == 'vtc':
            image_map = self._extract_media(record, media_type='image')
            if not image_map:
                raise ValueError('VTC mode requires at least one context image.')
            placeholders = ''.join(f'<image {index}>' for index in image_map)
            content_list.extend(self._parse_text_with_media(text=placeholders, image_map=image_map))
        elif self.eval_mode == 'text':
            context = _remove_html_tags(str(record['_context']))
            content_list.append(ContentText(text=context))
        else:
            raise ValueError(f'Unsupported eval_mode: {self.eval_mode}. Use "vtc" or "text".')

        content_list.append(ContentText(text=PROMPT_SUFFIXES[subset].format(question=problem)))
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=', '.join(answers),
            metadata={
                'problem': problem,
                'answers': answers,
                'subset': subset,
                'eval_mode': self.eval_mode,
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score a response with the official metric selected for its subset."""
        subset = task_state.metadata.get('subset')
        metrics = _calculate_metrics(original_prediction, task_state.metadata.get('answers', []))
        official_metric = 'rouge_l' if subset == 'Memory' else 'contains_all'
        return Score(
            extracted_prediction=_normalize_response(original_prediction),
            prediction=original_prediction,
            value={
                'score': metrics[official_metric],
                **metrics,
            },
            main_score_name='score',
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Expose the unified sample score as the canonical normalized report score."""
        aggregate_scores = super().aggregate_scores(sample_scores)
        for aggregate_score in aggregate_scores:
            if aggregate_score.metric_name != 'legacy_metric':
                continue
            if aggregate_score.dimensions.get('original_name') != 'score':
                continue
            aggregate_score.metric_name = 'normalized_score'
            aggregate_score.dimensions = {}
        return aggregate_scores
