import re
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    JudgeReview,
    OutputContract,
    Placement,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags

SPLIT_LIST = ['finance', 'legal', 'finance_hard', 'legal_hard']

DESCRIPTION = """
## Overview

PRBench (Professional Reasoning Benchmark) evaluates open-ended reasoning on realistic, high-stakes Finance and Legal
problems. Its expert-authored conversations and fine-grained rubrics measure whether a response is accurate, useful,
auditable, and appropriately handles uncertainty and risk.

## Task Description

- **Task Type**: Multi-turn open-ended question answering with rubric-based grading
- **Input**: One to ten conversation turns, optionally accompanied by reference texts
- **Output**: The assistant response to the final user turn
- **Domain**: Finance and Legal professional reasoning

## Key Features

- The current release contains 1,100 conversations and 18,692 expert-curated criteria across 13 Finance and 12 Legal
  topics; roughly 30% of the conversations are multi-turn.
- It covers 114 countries and dependencies and 47 U.S. jurisdictions, with both expert and non-expert user scenarios.
- Four dataset splits are available: `finance` (600), `legal` (500), `finance_hard` (300), and `legal_hard` (250). The
  hard splits contain the most difficult examples from their corresponding full splits.
- Each sample has 10–30 independently graded criteria with integer weights from -10 to 10 (excluding zero). Positive
  criteria describe desired properties, while negative criteria describe undesirable properties.

## Evaluation Notes

- Each rubric criterion is graded independently by an LLM judge as met or not met using the official prompt. The paper
  uses `o4-mini` as the judge; configure `judge.models` and use `judge.strategy='auto'` or `'llm'`.
- `clipped_score` is the primary metric reported by the paper: each sample's weighted points are divided by its total
  positive weight, then the mean across samples is clipped at a lower bound of 0.
- `normalized_score` subtracts each sample's minimum possible score (the sum of its negative weights), divides by the
  full score range, and then averages the sample scores. It is useful for comparisons across rubric distributions.
- Reference texts are prepended to their corresponding user turns exactly as in the official evaluator. Judge parse or
  transport failures exclude the affected sample rather than assigning a score.
- A full Finance plus Legal evaluation requires 18,692 judge calls per judge and repeat. The hard splits overlap the
  full splits, so EvalScope reports every split independently without an `OVERALL` row; select either the full or hard
  splits when publishing a combined result.
- Resources: [Paper](https://arxiv.org/abs/2511.11562) |
  [GitHub](https://github.com/scaleapi/PRBench) |
  [Dataset](https://modelscope.cn/datasets/ScaleAI/PRBench)
"""

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "The response refers to the need to confirm the business case for the workforce reduction proposed, e.g. is there a genuine reason for fewer employees, is the downturn likely to be long lasting, what reduction in head count is actually required?", and the response just mentions declines in demand, that would still meet the criteria even though it didn't mention one of the examples verbatim.

# Example 1
For example, if the conversation is "user: should I buy bitcoin? assistant: no just buy nvidia stock" and the rubric item is "tells the user to go to buy bitcoin", you should return a json like this:

```json
{
  "explanation": "The assistant does not tell the user to buy bitcoin.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: what are some red fruits? assistant: apples, strawberries" and the criterion is "fails to mention apples", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually mentions apples, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()  # noqa: E501

_THINKING_PATTERN = re.compile(r'<(think|thinking|reasoning|reason)>.*?</\1>', re.DOTALL | re.IGNORECASE)


class RubricGrade(BaseModel):
    """Binary result for one PRBench rubric criterion."""

    explanation: str = ''
    criteria_met: bool


RUBRIC_CONTRACT = OutputContract(schema_model=RubricGrade)


@register_benchmark(
    BenchmarkMeta(
        name='prbench',
        pretty_name='PRBench',
        dataset_id='ScaleAI/PRBench',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.REASONING, Tags.MULTI_TURN],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2511.11562',
        subset_list=SPLIT_LIST,
        default_subset='default',
        eval_split='finance',
        metric_list=['clipped_score', 'normalized_score'],
        primary_metric='clipped_score',
        aggregation='mean',
        prompt_template=None,
        evaluation_version='v1.0',
    )
)
class PRBenchAdapter(DefaultDataAdapter):
    """Adapter for PRBench's multi-turn conversations and weighted rubric judging."""

    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.split_as_subset = True
        self.add_overall_metric = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        messages: List[ChatMessage] = []
        for turn in range(10):
            prompt = record.get(f'prompt_{turn}')
            if isinstance(prompt, str) and prompt.strip():
                references = record.get(f'reference_texts_{turn}') or []
                reference_text = ''.join(
                    f'Reference Text {index}:\n{text}\n\n' for index, text in enumerate(references)
                )
                messages.append(ChatMessageUser(content=reference_text + prompt))

            response = record.get(f'response_{turn}')
            if isinstance(response, str) and response.strip():
                messages.append(ChatMessageAssistant(content=response))

        rubrics = [self._normalize_rubric(item) for item in record['rubric']]
        return Sample(
            input=messages,
            metadata={
                'task': record['task'],
                'field': record['field'],
                'topic': record['topic'],
                'expert': record['expert'],
                'turns': record['turns'],
                'rubrics': rubrics,
                'economic_pathway': record['economic_pathway'],
                'decision_type': record['decision_type'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        del task_state
        prediction = _THINKING_PATTERN.sub('', prediction)
        return re.sub(r'\n\s*\n\s*\n', '\n\n', prediction).strip()

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        rubrics = (context.task_state.metadata or {}).get('rubrics', [])
        cases = [
            JudgeCase(case_id=f'rubric_{index}', output_contract=RUBRIC_CONTRACT, metadata=rubric)
            for index, rubric in enumerate(rubrics)
        ]
        return JudgeDefinition.workflow(
            cases=cases,
            request=self._build_judge_request,
            reduce=self._reduce_verdicts,
            main_score_name='clipped_score',
            finalize=self._finalize_score,
        )

    @staticmethod
    def _normalize_rubric(rubric: Dict[str, Any]) -> Dict[str, Any]:
        annotations = rubric['annotations']
        weight_key = f'{annotations["weight_class"].replace(" ", "_")}_weight'
        weight = annotations.get(weight_key)
        if weight is None:
            raise ValueError(f'PRBench rubric {rubric.get("id", "unknown")} has no weight for {weight_key}.')
        return {
            'id': rubric['id'],
            'title': rubric['title'],
            'weight': float(weight),
            'category': annotations.get('criteria_category'),
        }

    @staticmethod
    def _build_judge_request(
        case: JudgeCase,
        placement: Placement,
        completed_cases: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> JudgeRequest:
        del placement, completed_cases
        messages = context.task_state.input
        if not isinstance(messages, list):
            raise ValueError('PRBench requires a chat-message input.')
        conversation = [*messages, ChatMessageAssistant(content=context.filtered_prediction)]
        conversation_text = '\n'.join(f'{message.role}: {message.text}' for message in conversation).strip()
        prompt = GRADER_TEMPLATE.replace('<<conversation>>', conversation_text).replace(
            '<<rubric_item>>', case.metadata['title']
        )
        return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

    @staticmethod
    def _reduce_verdicts(case_verdicts: Sequence[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
        del context
        weighted_points = sum(
            float(verdict.metadata['weight']) * float(verdict.value.criteria_met) for verdict in case_verdicts
        )
        positive_weight = sum(
            float(verdict.metadata['weight']) for verdict in case_verdicts if verdict.metadata['weight'] > 0
        )
        negative_weight = sum(
            float(verdict.metadata['weight']) for verdict in case_verdicts if verdict.metadata['weight'] < 0
        )
        if positive_weight <= 0:
            raise ValueError('PRBench requires at least one positive-weight rubric.')

        return ReducedVerdict(
            value={
                'clipped_score': weighted_points / positive_weight,
                'normalized_score': (weighted_points - negative_weight) / (positive_weight - negative_weight),
            },
            metadata={
                'weighted_points': weighted_points,
                'positive_weight': positive_weight,
                'negative_weight': negative_weight,
                'criteria_met': sum(bool(verdict.value.criteria_met) for verdict in case_verdicts),
                'rubric_count': len(case_verdicts),
            },
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Apply the official aggregation independently to each PRBench metric."""
        aggregated = super().aggregate_scores(sample_scores)
        for score in aggregated:
            if score.metric_name != 'clipped_score':
                continue
            score.score = max(0.0, score.score)
            score.aggregation = 'clipped_mean'
        return aggregated

    @staticmethod
    def _finalize_score(score: Score, review: JudgeReview, context: JudgeContext) -> Score:
        del context
        score.explanation = (
            f'{review.metadata.get("criteria_met", 0)}/{review.metadata.get("rubric_count", 0)} criteria met; '
            f'weighted points {review.metadata.get("weighted_points", 0):g}/'
            f'{review.metadata.get("positive_weight", 0):g}.'
        )
        return score
