# flake8: noqa: E501
from pydantic import BaseModel, Field, create_model
from typing import Any, Dict, List, Type

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
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

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        metadata = context.task_state.metadata or {}
        if not metadata.get('components', []):
            logger.warning('No components found in sample metadata; assigning zero score.')
            return JudgeDefinition.skip(
                Score(
                    extracted_prediction=context.filtered_prediction,
                    prediction=context.original_prediction,
                    value={'judge_score': 0.0},
                    main_score_name='judge_score'
                ),
                reason='missing_components',
            )
        contract = OutputContract(schema_model=_build_grade_schema(metadata.get('component_weight', [])))

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            from .utils import generate_mia_judge_prompt
            current = judge_context.task_state.metadata or {}
            prompt = generate_mia_judge_prompt(
                instruction=current.get('instruction', judge_context.task_state.input_text),
                components=current.get('components', []),
                component_weight=current.get('component_weight', []),
                response=judge_context.filtered_prediction or judge_context.original_prediction,
            ) + case.output_contract.instruction()
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            grade, current = case_verdicts[0].value, judge_context.task_state.metadata or {}
            values, raw_sum = {}, 0.0
            weights = current.get('component_weight', [])
            for index, component_type in enumerate(current.get('component_type', [])):
                raw = float(getattr(grade, f'component_{index + 1}'))
                weight = weights[index] if index < len(weights) else 1
                values[f'component_{index + 1}_{component_type}'] = raw / weight if weight else 0.0
                raw_sum += raw
            values['judge_score'] = raw_sum / sum(weights) if sum(weights) else 0.0
            return ReducedVerdict(value=values)

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='grade', output_contract=contract)],
            request=request,
            reduce=reduce,
            main_score_name='judge_score'
        )
