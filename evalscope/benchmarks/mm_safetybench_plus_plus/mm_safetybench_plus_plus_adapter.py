# flake8: noqa: E501
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = """
## Overview

MM-SafetyBench++ is a multimodal contextual-safety benchmark (arXiv:2603.15800) that evaluates
whether MLLMs adapt their safety behavior to user intent. Each unsafe image-text pair is paired
with a minimally edited safe counterpart, so models must distinguish subtly different intents
that share the same visual context rather than rely on keyword triggers.

## Task Description

- **Task Type**: Contextual safety evaluation (free-form response)
- **Input**: Synthetic scenario image + user query (unsafe or minimally edited safe variant)
- **Output**: Free-form text response
- **Domain**: Safety (illegal activity, hate speech, malware generation, physical harm, fraud, sex)

## Key Features

- 2,844 test samples: 6 harm categories x safe/unsafe pairs x 2 image attack modes (Gen, GenOCR)
- Balanced design: 1,422 unsafe queries and 1,422 minimally edited safe counterparts
- High-fidelity synthetic images with query-relevant context, improving on MM-SafetyBench
- Evaluation via the official two-branch RQ-5 rubric (0-5) with an LLM judge

## Evaluation Notes

- Primary metric: **quality_score**, the mean RQ-5 rating (0-5) assigned by the judge
- Unsafe queries use the official safety rubric (risk recognition, refusal quality, safe alternatives);
  safe queries use the official helpfulness rubric (intent understanding, accuracy, specificity)
- Requires an LLM judge configured through `judge.models`
- Subsets: `unsafe` and `safe`; category and image mode are recorded in sample metadata
"""

SUBSET_LIST = ['unsafe', 'safe']


class QualityGrade(BaseModel):
    """Reply key required from the RQ-5 judge."""

    score: int = Field(ge=0, le=5)


QUALITY_CONTRACT = OutputContract(schema_model=QualityGrade)


@register_benchmark(
    BenchmarkMeta(
        name='mm_safetybench_plus_plus',
        pretty_name='MM-SafetyBench++',
        dataset_id='evalscope/MM-SafetyBench-plus-plus',
        hf_dataset_id='EchoSafe-MLLM/MM-SafetyBench-plus-plus',
        tags=[Tags.MULTI_MODAL, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2603.15800',
        subset_list=SUBSET_LIST,
        metric_list=['quality_score'],
        eval_split='test',
        evaluation_version='v1.0',
    )
)
class MMSafetyBenchPlusPlusAdapter(VisionLanguageAdapter):
    """Data adapter for MM-SafetyBench++ contextual-safety evaluation.

    Each record is one image-text query from the unsafe subset or its minimally edited
    safe counterpart. Scoring uses the official RQ-5 rubric through an LLM judge, with
    the rubric branch selected by the record label.
    """

    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.save_metadata = True

    def record_to_sample(self, record: Dict[str, Any]) -> Optional[Sample]:
        label = record.get('label', '')
        if label not in ('unsafe', 'safe'):
            logger.warning(f'Record with unknown label {label!r}; skipping.')
            return None

        image = record.get('image')
        if image is None:
            logger.warning('Record missing image field, skipping.')
            return None

        image_b64 = self._image_bytes_to_base64(image['bytes'], default_format='png')

        return Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentImage(image=image_b64),
                        ContentText(text=record['question']),
                    ]
                )
            ],
            target='',
            subset_key=label,
            metadata={
                'label': label,
                'category': record.get('category', ''),
                'mode': record.get('mode', ''),
            },
        )

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        from .utils import build_safe_judge_prompt, build_unsafe_judge_prompt

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            metadata = judge_context.task_state.metadata or {}
            if metadata.get('label') == 'unsafe':
                prompt = build_unsafe_judge_prompt(judge_context.original_prediction)
            else:
                prompt = build_safe_judge_prompt(judge_context.original_prediction)
            return JudgeRequest(messages=[ChatMessageUser(content=prompt + QUALITY_CONTRACT.instruction())])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            return ReducedVerdict(value={'quality_score': float(case_verdicts[0].value.score)})

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='quality', output_contract=QUALITY_CONTRACT)],
            request=request,
            reduce=reduce,
            main_score_name='quality_score',
        )
