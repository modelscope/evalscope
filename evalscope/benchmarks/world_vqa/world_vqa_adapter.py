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

SUBSET_LIST = [
    'Nature & Environment',
    'Locations & Architecture',
    'Culture, Arts & Crafts',
    'Objects & Products',
    'Vehicles, Craft & Transportation',
    'Entertainment, Media & Gaming',
    'Brands, Logos & Graphic Design',
    'Sports, Gear & Venues',
]

PROMPT_TEMPLATE = """{question}"""


@register_benchmark(
    BenchmarkMeta(
        name='world_vqa',
        pretty_name='WorldVQA',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

WorldVQA is a benchmark designed to evaluate the atomic visual world knowledge of Multimodal Large Language Models (MLLMs). It measures models' ability to ground and name visual entities across a stratified taxonomy, spanning from common head-class objects to long-tail rarities.

## Task Description

- **Task Type**: Visual Entity Recognition / Knowledge QA
- **Input**: Image + question asking to identify a visual entity
- **Output**: Free-form text answer (specific entity name)
- **Domain**: Nature, architecture, culture, products, transportation, entertainment, brands, sports

## Key Features

- 3000 VQA pairs across 8 semantic categories
- Bilingual: English (non-zh) and Chinese (zh)
- Three difficulty levels: easy, medium, hard
- Tests atomic visual knowledge decoupled from reasoning
- Requires precise entity identification (e.g., specific breed, not generic "dog")

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on **train** split (the benchmark data split)
- Primary metric: **Accuracy** via LLM-as-judge
- Supports LLM judge for semantic equivalence checking
- Results reported per category and overall
""",
        dataset_id='evalscope/WorldVQA',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='train',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class WorldVQAAdapter(VisionLanguageAdapter):

    llm_judge_default = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # WorldVQA requires LLM judge for semantic equivalence evaluation
        self.reformat_subset = True
        self.save_metadata = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record.get('question', '')
        answer = record.get('answer', '')

        content_list: List[Content] = [ContentText(text=question)]

        # Image is stored as base64 string in the TSV
        image_data = record.get('image', '')
        if image_data:
            # Add data URI header for PNG
            image_base64 = f'data:image/png;base64,{image_data}'
            content_list.append(ContentImage(image=image_base64))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=answer,
            subset_key=record.get('category'),
            metadata={
                'index': record.get('index'),
                'category': record.get('category'),
                'language': record.get('language'),
                'difficulty': record.get('difficulty'),
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Rule-based scoring: exact match (case-insensitive)."""
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        # Case-insensitive exact match as baseline
        result = 1 if filtered_prediction.strip().lower() == reference.strip().lower() else 0
        score.value = {'acc': result}
        return score
