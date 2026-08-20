from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeRequest,
    OutputContract,
    PairwiseOutcome,
    PairwisePlacementOutcome,
    Placement,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


# The judge replies with the single letter of the better output; 'm' is the baseline and 'M'
# the evaluated model. Case matters, so an unanchored search would match any m in prose.
class Preference(BaseModel):
    verdict: Literal['m', 'M']


PREFERENCE_CONTRACT = OutputContract(schema_model=Preference)

GRADER_SYSTEM_PROMPT = """You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."""  # noqa: E501

GRADER_TEMPLATE = """
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": "{instruction}"
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "m",
        "output": "{output_1}"
    }},
    {{
        "model_identifier": "M",
        "output": "{output_2}"
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier
""".strip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='alpaca_eval',
        pretty_name='AlpacaEval2.0',
        tags=[Tags.INSTRUCTION_FOLLOWING, Tags.ARENA],
        description="""
## Overview

AlpacaEval 2.0 is an evaluation framework for instruction-following language models that uses an LLM judge to compare model outputs against a strong baseline. It provides win-rate metrics reflecting human preferences.

## Task Description

- **Task Type**: Instruction-Following Evaluation (Pairwise Comparison)
- **Input**: User instruction/question
- **Output**: Model response compared against GPT-4 Turbo baseline
- **Metric**: Win rate against baseline model

## Key Features

- Auto-annotator for scalable evaluation
- Compares against GPT-4 Turbo baseline outputs
- High correlation with human preferences
- Cost-effective evaluation method
- Tests general instruction-following capabilities

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM judge (default: gpt-4-1106-preview)
- Baseline model: gpt-4-turbo outputs
- Reports win rate metric
- Note: Length-controlled win rate not currently supported
""",
        dataset_id='AI-ModelScope/alpaca_eval',
        subset_list=['alpaca_eval_gpt4_baseline'],
        metric_list=['win_rate'],
        few_shot_num=0,
        train_split=None,
        eval_split='eval',
        prompt_template='{question}'
    )
)
class AlpacaEvalAdapter(DefaultDataAdapter):

    scoring_policy = ScoringPolicy.JUDGE_ONLY
    judge_revision = '2'
    uses_pairwise_outcome = True
    supports_position_swap = True
    official_position_swap = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        instruction = record['instruction']
        baseline_output = record['output']  # baseline model output

        return Sample(
            input=instruction,
            target=baseline_output,
            metadata={
                'generator': record.get('generator', 'unknown'),
                'dataset': record.get('dataset', 'unknown')
            }
        )

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        return [JudgeCase(case_id='preference', output_contract=PREFERENCE_CONTRACT)]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        # The official labels name output slots, so a swapped pass also reverses the label map.
        candidate_first = placement is Placement.SWAPPED
        prompt = GRADER_TEMPLATE.format(
            instruction=context.task_state.input_text,
            output_1=context.filtered_prediction if candidate_first else context.reference,
            output_2=context.reference if candidate_first else context.filtered_prediction,
        )
        prompt += case.output_contract.instruction()
        return JudgeRequest(messages=[ChatMessageSystem(content=GRADER_SYSTEM_PROMPT), ChatMessageUser(content=prompt)])

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        placements = case_verdicts[0].placements
        values = placements or {'original': case_verdicts[0].value}
        outcomes = {
            name: PairwisePlacementOutcome(
                result='win' if verdict.verdict == ('m' if name == 'swapped' else 'M') else 'loss'
            )
            for name, verdict in values.items()
        }
        result = next(iter(outcome.result for outcome in outcomes.values())) \
            if len({outcome.result for outcome in outcomes.values()}) == 1 else 'tie'
        outcome = PairwiseOutcome(metric_name='win_rate', result=result, placements=outcomes)
        return ReducedVerdict(value={'win_rate': outcome.score}, outcome=outcome)

    def finalize_judge_score(self, review, context) -> Score:
        score = super().finalize_judge_score(review, context)
        score.main_score_name = 'win_rate'
        return score
