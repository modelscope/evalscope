from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt

logger = get_logger()

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT


@register_benchmark(
    BenchmarkMeta(
        name='cc_bench',
        pretty_name='CCBench',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

CCBench (Chinese Culture Bench) is an extension of MMBench specifically designed to evaluate multimodal models' understanding of Chinese traditional culture. It covers various aspects of Chinese cultural heritage through visual question answering.

## Task Description

- **Task Type**: Visual Multiple-Choice Q&A (Chinese Culture)
- **Input**: Image with question about Chinese culture
- **Output**: Single correct answer letter (A, B, C, or D)
- **Language**: Primarily Chinese content

## Key Features

- Questions about Chinese traditional culture
- Categories: Calligraphy, Painting, Cultural Relics, Food & Clothes
- Historical Figures, Scenery & Building, Sketch Reasoning, Traditional Shows
- Tests cultural knowledge combined with visual understanding
- Extension of the MMBench evaluation framework

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses Chain-of-Thought (CoT) prompting
- Evaluates on test split
- Simple accuracy metric for scoring
- Requires both visual perception and cultural knowledge
""",
        dataset_id='lmms-lab/MMBench',
        subset_list=['cc'],
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class CCBenchAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answers_list: List[str] = [record.get('A', ''), record.get('B', ''), record.get('C', ''), record.get('D', '')]
        input_text = prompt(question=record['question'], choices=answers_list, template=self.prompt_template)
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        label_answer = record.get('answer')
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=label_answer,
            metadata={
                'index': record.get('index'),
                'category': record.get('category'),
                'source': record.get('source')
            }
        )


@register_benchmark(
    BenchmarkMeta(
        name='mm_bench',
        pretty_name='MMBench',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

MMBench is a systematically designed benchmark for evaluating vision-language models across 20 fine-grained ability dimensions. It uses a novel CircularEval strategy and provides both English and Chinese versions for cross-lingual evaluation.

## Task Description

- **Task Type**: Visual Multiple-Choice Q&A
- **Input**: Image with question and 2-4 answer options
- **Output**: Single correct answer letter (A, B, C, or D)
- **Languages**: English (en) and Chinese (cn) subsets

## Key Features

- 20 fine-grained ability dimensions defined
- Systematic evaluation pipeline with CircularEval strategy
- Bilingual support (English and Chinese)
- Questions include hints for context
- Tests perception, reasoning, and knowledge abilities

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses Chain-of-Thought (CoT) prompting
- Evaluates on dev split
- Two subsets: `cn` (Chinese) and `en` (English)
- Results include category-level breakdown
""",
        dataset_id='lmms-lab/MMBench',
        subset_list=['cn', 'en'],
        metric_list=['acc'],
        eval_split='dev',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class MMBenchAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answers_list: List[str] = [record.get('A', ''), record.get('B', ''), record.get('C', ''), record.get('D', '')]
        answers_list = [ans for ans in answers_list if (ans.strip() and ans != 'nan')]
        question_hint = record['hint'] + record['question']
        input_text = prompt(question=question_hint, choices=answers_list, template=self.prompt_template)
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        label_answer = record.get('answer')
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=label_answer,
            metadata={
                'index': record.get('index'),
                'category': record.get('category'),
                'source': record.get('source'),
                'L2-category': record.get('L2-category'),
                'comment': record.get('comment'),
                'split': record.get('split')
            }
        )
