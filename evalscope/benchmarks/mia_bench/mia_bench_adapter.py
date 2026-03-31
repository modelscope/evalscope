# flake8: noqa: E501
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()


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
- Primary metric: **total_score** (mean of per-sample normalized 0–1 total scores)
- Requires a capable LLM judge (e.g., GPT-4o, Qwen-Max) configured via `judge_model_args`
- Judge strategy should be set to `JudgeStrategy.LLM`
""",
        metric_list=['total_score'],
        eval_split='test',
    )
)
class MIABenchAdapter(VisionLanguageAdapter):
    """
    Data adapter for lmms-lab/MIA-Bench.

    Each sample is scored by an LLM judge that evaluates whether the model's
    response satisfies each weighted instruction component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_llm_judge = True  # Always use LLM judge for this benchmark
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

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """
        Use an LLM judge to score the model's response against MIA-Bench components.

        The judge is asked to rate each instruction component separately and
        provide a total score. Scores are parsed and normalised to [0, 1].
        """
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        metadata = task_state.metadata or {}
        instruction: str = metadata.get('instruction', task_state.input_text)
        components: List[str] = metadata.get('components', [])
        component_weight: List[int] = metadata.get('component_weight', [])
        component_type: List[str] = metadata.get('component_type', [])

        if not components:
            logger.warning('No components found in sample metadata; assigning zero score.')
            score.value = {'total_score': 0.0}
            score.main_score_name = 'total_score'
            return score

        # Build judge prompt
        from .utils import generate_mia_judge_prompt, parse_mia_score

        judge_prompt = generate_mia_judge_prompt(
            instruction=instruction,
            components=components,
            component_weight=component_weight,
            response=filtered_prediction or original_prediction,
        )

        # Call LLM judge
        judge_response = self.llm_judge.judge(judge_prompt)

        # Parse scores
        score_dict = parse_mia_score(component_type, judge_response)

        # Build score.value: total_score + per-component scores with unique keys
        score_value: Dict[str, float] = {'total_score': score_dict.get('total_score', 0.0)}
        for i, ctype in enumerate(component_type):
            key = f'component_{i + 1}_{ctype}'
            score_value[key] = score_dict.get(ctype, 0.0)

        score.value = score_value
        score.main_score_name = 'total_score'
        score.explanation = f'LLM judge response:\n{judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
            'components': components,
            'component_weight': component_weight,
            'component_type': component_type,
        }

        return score
