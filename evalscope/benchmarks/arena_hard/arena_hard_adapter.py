# flake8: noqa: E501
from pydantic import BaseModel
from typing import Any, Dict, List, Literal

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import (
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    PairwiseOutcome,
    PairwisePlacementOutcome,
    Placement,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


class BattleVerdict(BaseModel):
    """One game's verdict on the official five-point preference scale."""
    reasoning: str = ''
    verdict: Literal['A>>B', 'A>B', 'A=B', 'B>A', 'B>>A']


BATTLE_CONTRACT = OutputContract(schema_model=BattleVerdict)

GRADER_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must state your final verdict as one of: A>>B (Assistant A is significantly better), A>B (Assistant A is slightly better), A=B (tie), B>A (Assistant B is slightly better), or B>>A (Assistant B is significantly better)."""  # noqa: E501

GRADER_TEMPLATE = """<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>""".strip(
)


@register_benchmark(
    BenchmarkMeta(
        name='arena_hard',
        pretty_name='ArenaHard',
        tags=[Tags.INSTRUCTION_FOLLOWING, Tags.ARENA],
        description="""
## Overview

ArenaHard is a challenging benchmark that evaluates language models through competitive pairwise comparison. Models are judged against a GPT-4 baseline on difficult tasks requiring reasoning, understanding, and generation capabilities.

## Task Description

- **Task Type**: Competitive Model Evaluation (Arena-style)
- **Input**: Challenging instruction/question
- **Output**: Model response compared against GPT-4-0314 baseline
- **Scoring**: Elo-based rating from pairwise battles

## Key Features

- 500 challenging user prompts
- Two-game battle system (A vs B and B vs A)
- Elo rating calculation for model ranking
- Tests reasoning, instruction-following, and generation
- High correlation with Chatbot Arena rankings

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM judge (default: gpt-4-1106-preview)
- Baseline model: gpt-4-0314 outputs
- Reports win rate and Elo-based scores
- Note: Style-controlled win rate not currently supported
""",
        dataset_id='AI-ModelScope/arena-hard-auto-v0.1',
        metric_list=['win_rate'],
        aggregation='elo',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}'
    )
)
class ArenaHardAdapter(DefaultDataAdapter):

    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        check_import(module_name=['sklearn'], extra='arena_hard', raise_error=True, feature_name=self.pretty_name)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        question = record['question']
        baseline_prediction = record['prediction']  # baseline model prediction

        return Sample(
            input=question, target=baseline_prediction, metadata={'capability': record.get('capability', 'unknown')}
        )

    supports_position_swap = True
    official_position_swap = True
    """Arena-Hard plays each pair twice with the answers swapped."""

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            baseline_first = placement is Placement.ORIGINAL
            prompt = GRADER_TEMPLATE.format(
                question=judge_context.task_state.input_text,
                answer_1=judge_context.reference if baseline_first else judge_context.filtered_prediction,
                answer_2=judge_context.filtered_prediction if baseline_first else judge_context.reference,
            )
            return JudgeRequest(
                messages=[
                    ChatMessageSystem(content=GRADER_SYSTEM_PROMPT),
                    ChatMessageUser(content=prompt + case.output_contract.instruction())
                ]
            )

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            placements = case_verdicts[0].placements
            res1 = placements.get('original', case_verdicts[0].value).verdict
            res2 = placements.get('swapped')
            outcomes = {'original': _placement_outcome(res1, candidate_is_a=False)}
            if res2 is not None:
                outcomes['swapped'] = _placement_outcome(res2.verdict, candidate_is_a=True)
            result, strength = _reduce_placements(outcomes)
            outcome = PairwiseOutcome(metric_name='score', result=result, strength=strength, placements=outcomes)
            return ReducedVerdict(value={'score': outcome.score}, outcome=outcome)

        def finalize(score, review, judge_context) -> Score:
            if review.outcome is not None:
                score.metadata['battle_result'] = {
                    'model_a': 'gpt4-0314',
                    'model_b': 'test_model',
                    'games': [{
                        'score': _battle_label(review.outcome.placements['original'], candidate_is_a=False)
                    }, *([{
                        'score': _battle_label(review.outcome.placements['swapped'], candidate_is_a=True)
                    }] if 'swapped' in review.outcome.placements else [])],
                }
            return score

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='battle', output_contract=BATTLE_CONTRACT)],
            request=request,
            reduce=reduce,
            main_score_name='score',
            finalize=finalize,
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        import pandas as pd

        from .utils import compute_mle_elo, get_battles_from_row, get_bootstrap_result, get_win_rate_column

        # A sample whose judge verdicts were unusable has no battle to contribute.
        scored = [res for res in sample_scores if (res.score.metadata or {}).get('battle_result')]
        if not scored:
            return []
        battles = pd.concat([get_battles_from_row(res.score.metadata['battle_result']) for res in scored])

        bootstrap_online_elo = compute_mle_elo(battles)

        stats = pd.DataFrame()
        stats['results'] = None
        stats['results'] = stats['results'].astype('object')

        for i, model in enumerate(bootstrap_online_elo.index):
            # assert model in bootstrap_elo_lu.columns
            stats.at[i, 'model'] = model
            stats.at[i, 'score'] = bootstrap_online_elo[model]

        score = get_win_rate_column(stats, 'score', 'gpt4-0314').at['test_model']

        return [AggScore(
            score=score,
            metric_name='win_rate',
            num=len(scored),
        )]


def _candidate_outcome(label: str, candidate_is_a: bool) -> str:
    if label == 'A=B':
        return 'tie'
    return 'win' if label.startswith('A') == candidate_is_a else 'loss'


def _placement_outcome(label: str, candidate_is_a: bool) -> PairwisePlacementOutcome:
    return PairwisePlacementOutcome(
        result=_candidate_outcome(label, candidate_is_a),
        strength='strong' if label in ('A>>B', 'B>>A') else 'weak',
    )


def _reduce_placements(placements: Dict[str, PairwisePlacementOutcome]) -> tuple[str, str]:
    results = [placement.result for placement in placements.values()]
    result = results[0] if len(set(results)) == 1 else 'tie'
    strength = 'strong' if result != 'tie' and any(
        placement.result == result and placement.strength == 'strong' for placement in placements.values()
    ) else 'weak'
    return result, strength


def _battle_label(outcome: PairwisePlacementOutcome, candidate_is_a: bool) -> str:
    if outcome.result == 'tie':
        return 'A=B'
    a_wins = (outcome.result == 'win') == candidate_is_a
    marker = '>>' if outcome.strength == 'strong' else '>'
    return f'A{marker}B' if a_wins else f'B{marker}A'
