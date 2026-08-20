# flake8: noqa: E501
from pydantic import BaseModel, Field, create_model
from typing import Any, Dict, List, Type

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()


def _build_grade_schema(component_weight: List[int]) -> Type[BaseModel]:
    """A per-sample schema: ``component_i`` bounded by that component's weight, plus reasoning."""
    fields: Dict[str, Any] = {'reasoning': (str, Field(default=''))}
    for i, weight in enumerate(component_weight):
        fields[f'component_{i + 1}'] = (float, Field(ge=0.0, le=float(weight)))
    return create_model('MiaGrade', **fields)


@register_benchmark(
    BenchmarkMeta(
        name='mia_bench',
        pretty_name='MIA-Bench',
        dataset_id='lmms-lab/MIA-Bench',
        tags=[Tags.MULTI_MODAL, Tags.INSTRUCTION_FOLLOWING, Tags.QA],
        description="""
## Overview

MIA-Bench is a multimodal instruction-following benchmark designed to evaluate vision-language models on their ability to follow complex, compositional instructions grounded in images. Each sample contains an image paired with a multi-component instruction, and model responses are scored by an LLM judge per component.

## Task Description

- **Task Type**: Multimodal Instruction Following
- **Input**: Image + multi-component instruction
- **Output**: Free-form response following all instruction components
- **Domains**: Visual understanding, instruction following, language generation

## Key Features

- 400 test samples with diverse instruction types (basic to advanced)
- Each instruction decomposes into 1–5 graded components with weighted scores
- Component types include: describe, length_limit, linguistics, format, etc.
- LLM-as-judge scoring: judge evaluates each component independently and gives a weighted total score (0–10 range, normalized to 0–1)
- No predefined reference answers; scoring is fully judge-based

## Evaluation Notes

- Default evaluation uses the **test** split (400 samples)
- Primary metric: **judge_score** (mean of per-sample normalized 0–1 total scores)
- Requires a capable LLM judge (e.g., GPT-4o, Qwen-Max) configured through `judge.models`
- Judge strategy should be set to `judge.strategy='llm'`
""",
        metric_list=['judge_score'],
        eval_split='test',
    )
)
class MIABenchAdapter(VisionLanguageAdapter):
    """
    Data adapter for lmms-lab/MIA-Bench.

    Each sample is scored by an LLM judge that evaluates whether the model's
    response satisfies each weighted instruction component.
    """
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_metadata = False  # Metadata (PIL images etc.) should not be serialised

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a raw MIA-Bench record to a Sample."""
        image_bytes = record['image']['bytes']
        image_b64 = bytes_to_base64(image_bytes, format='jpeg', add_header=True)

        instruction = record['instruction']
        content_list: List[Content] = [
            ContentImage(image=image_b64),
            ContentText(text=instruction),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target='',  # No ground-truth answer; scoring is judge-based
            metadata={
                'instruction': instruction,
                'type': record.get('type', ''),
                'num_of_component': record.get('num_of_component', len(record.get('components', []))),
                'components': record.get('components', []),
                'component_weight': record.get('component_weight', []),
                'component_type': record.get('component_type', []),
            },
        )

    def pre_judge_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        components = (task_state.metadata or {}).get('components', [])
        if not components:
            logger.warning('No components found in sample metadata; assigning zero score.')
            return Score(
                extracted_prediction=filtered_prediction,
                prediction=original_prediction,
                value={'judge_score': 0.0},
                main_score_name='judge_score',
            )
        return None

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        # The schema is per-sample: one bounded field per instruction component, so a raw score
        # above a component's weight is a parse failure rather than a silently clamped value.
        metadata = context.task_state.metadata or {}
        contract = OutputContract(schema_model=_build_grade_schema(metadata.get('component_weight', [])))
        return [JudgeCase(case_id='grade', output_contract=contract)]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        from .utils import generate_mia_judge_prompt

        metadata = context.task_state.metadata or {}
        prompt = generate_mia_judge_prompt(
            instruction=metadata.get('instruction', context.task_state.input_text),
            components=metadata.get('components', []),
            component_weight=metadata.get('component_weight', []),
            response=context.filtered_prediction or context.original_prediction,
        )
        prompt += case.output_contract.instruction()
        return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        grade = case_verdicts[0].value
        metadata = context.task_state.metadata or {}
        component_type: List[str] = metadata.get('component_type', [])
        weights: List[int] = metadata.get('component_weight', [])

        value: Dict[str, float] = {}
        raw_sum = 0.0
        for i, ctype in enumerate(component_type):
            raw = float(getattr(grade, f'component_{i + 1}'))
            weight = weights[i] if i < len(weights) else 1
            value[f'component_{i + 1}_{ctype}'] = raw / weight if weight else 0.0
            raw_sum += raw
        # `judge_score` is derived from the components, the single source of truth, rather than
        # trusting a separate total the judge might miscompute.
        total_weight = sum(weights)
        value['judge_score'] = raw_sum / total_weight if total_weight else 0.0
        return ReducedVerdict(value=value)

    def finalize_judge_score(self, review, context) -> Score:
        score = super().finalize_judge_score(review, context)
        score.main_score_name = 'judge_score'
        return score
