# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, ImageEditAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessage, ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import FileConstants, ScoringPolicy, Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    'background_change', 'color_alter', 'material_alter', 'motion_change', 'ps_human', 'style_change', 'subject-add',
    'subject-remove', 'subject-replace', 'text_change', 'tone_transfer'
]

LANGUAGE_LIST = ['en', 'cn']


class GeditGrade(BaseModel):
    score: List[int] = Field(min_length=1)
    reasoning: str = ''


GEDIT_CONTRACT = OutputContract(schema_model=GeditGrade)


@register_benchmark(
    BenchmarkMeta(
        name='gedit',
        pretty_name='GEdit-Bench',
        dataset_id='stepfun-ai/GEdit-Bench',
        description="""
## Overview

GEdit-Bench (Grounded Edit Benchmark) is an image editing benchmark grounded in real-world usage scenarios. It provides comprehensive evaluation of image editing models across diverse editing tasks with LLM-based judging.

## Task Description

- **Task Type**: Image Editing Evaluation
- **Input**: Source image + editing instruction
- **Output**: Edited image evaluated by LLM judge
- **Languages**: English (en) and Chinese (cn)

## Key Features

- Real-world editing scenarios (background change, color alter, style transfer, etc.)
- 11 editing task categories
- LLM-based evaluation for semantic consistency and perceptual quality
- Supports both English and Chinese instructions
- Comprehensive scoring: `semantic_consistency`, `perceptual_similarity`, `normalized_score`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on **train** split (contains test samples)
- Metrics: **semantic_consistency**, **perceptual_similarity** (via LLM judge)
- `normalized_score` is the official Overall: the geometric mean of the SC and PQ scores
- Configure language via `extra_params['language']` (en/cn)
""",
        tags=[Tags.IMAGE_EDITING],
        subset_list=SUBSET_LIST,
        metric_list=['semantic_consistency', 'perceptual_similarity', 'normalized_score'],
        primary_metric='normalized_score',
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        extra_params={
            'language': {
                'type': 'str',
                'description': f'Language of the instruction. Choices: {LANGUAGE_LIST}.',
                'value': 'en',
                'choices': LANGUAGE_LIST
            }
        }
    )
)
class GEditAdapter(ImageEditAdapter):

    scoring_policy = ScoringPolicy.JUDGE_ONLY
    judge_revision = '1'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.language = self.extra_params.get('language', 'en')
        self.reformat_subset = True

        self.load_prompt()

    def load_prompt(self):
        from . import vie_prompts

        self.context = vie_prompts._context_no_delimit
        self.SC_prompt = '\n'.join([
            self.context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC
        ])
        self.PQ_prompt = '\n'.join([self.context, vie_prompts._prompts_0shot_rule_PQ])

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        record = copy.deepcopy(record)

        # Process instruction and image
        instruction = record['instruction']
        image_bytes = record['input_image']['bytes']
        input_image = bytes_to_base64(image_bytes, format='png', add_header=True)
        record['input_image'] = input_image
        record[FileConstants.ID] = record['key']
        del record['input_image_raw']

        text_content = ContentText(text=instruction)
        image_content = ContentImage(image=input_image)

        messages: List[ChatMessage] = [
            ChatMessageUser(content=[text_content, image_content]),
        ]

        return Sample(input=messages, subset_key=record['task_type'], metadata=record)

    def sample_filter(self, sample: Sample) -> bool:
        language = sample.metadata.get('instruction_language', 'en')
        return super().sample_filter(sample) and language == self.language

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        return [
            JudgeCase(case_id='SC', output_contract=GEDIT_CONTRACT, metadata={'kind': 'SC'}),
            JudgeCase(case_id='PQ', output_contract=GEDIT_CONTRACT, metadata={'kind': 'PQ'}),
        ]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        metadata = context.task_state.metadata or {}
        edited_image = metadata[FileConstants.IMAGE_PATH]
        if case.metadata['kind'] == 'SC':
            input_image = metadata['input_image']
            text = self.SC_prompt.replace('<instruction>', metadata['instruction'])
            content = [ContentImage(image=input_image), ContentImage(image=edited_image), ContentText(text=text)]
        else:
            content = [ContentImage(image=edited_image), ContentText(text=self.PQ_prompt)]
        prompt_text = content[-1].text + case.output_contract.instruction()
        content[-1] = ContentText(text=prompt_text)
        return JudgeRequest(messages=[ChatMessageUser(content=content)])

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        import math

        verdicts_by_case = {verdict.case_id: verdict for verdict in case_verdicts}
        sc_score = min(verdicts_by_case['SC'].value.score)
        pq_score = min(verdicts_by_case['PQ'].value.score)
        return ReducedVerdict(
            value={
                'semantic_consistency': float(sc_score),
                'perceptual_similarity': float(pq_score),
                'normalized_score': math.sqrt(sc_score * pq_score),
            }
        )

    def finalize_judge_score(self, review, context) -> Score:
        score = super().finalize_judge_score(review, context)
        score.main_score_name = 'normalized_score'
        metadata = context.task_state.metadata or {}
        score.extracted_prediction = metadata.get(FileConstants.IMAGE_PATH, '')
        score.prediction = score.extracted_prediction
        return score
