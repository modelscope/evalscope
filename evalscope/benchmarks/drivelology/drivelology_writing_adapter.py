from pydantic import BaseModel, Field
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageUser, ContentText
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


class NarrativeRating(BaseModel):
    reasoning: str = ''
    rating: int = Field(ge=1, le=5)


RATING_CONTRACT = OutputContract(schema_model=NarrativeRating)

DESCRIPTION = """
## Overview

Drivelology Narrative Writing evaluates models' ability to generate detailed descriptions illustrating the implicit narrative of "drivelology" text - linguistic utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.

## Task Description

- **Task Type**: Narrative Generation and Evaluation
- **Input**: Drivelology text sample
- **Output**: Generated narrative description explaining implicit meaning
- **Domain**: Linguistic analysis, narrative generation

## Key Features

- Tests narrative explanation generation ability
- Requires understanding of layered linguistic meanings
- LLM-as-judge evaluation against reference narratives
- Likert scale scoring (1-5) for match quality
- Tests depth of linguistic and cultural understanding

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM-as-judge for evaluation
- Metrics: Average Likert score (1-5 scale)
- Evaluates relevance, accuracy, depth, and detail of generated narratives
"""

# Keep the original generation and evaluation templates
NARRATIVE_GENERATION_TEMPLATE = """
You need to first read and understand the text given. Generate a detailed description to illustrate the implicit narrative of the text.

Please provide your response in English, with a clear and comprehensive explanation of the narrative.

Text: {text}
""".strip()  # noqa: E501

NARRATIVE_EVALUATION_TEMPLATE = """
Please act as an impartial judge and evaluate how accurately the candidate narrative matches the given reference narrative.
Your evaluation should consider factors such as the relevance, accuracy, depth, and level of detail of the candidate narrative compared to the reference.

Begin your evaluation by providing a short explanation in English. Be as objective as possible.

After providing your explanation, you must rate the match on a Likert scale from 1 to 5, where:
1 = Very poor match
2 = Poor match
3 = Moderate match
4 = Good match
5 = Excellent match


[Candidate Narrative]
{candidate}

[Reference Narrative]
{reference}
""".strip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='drivel_writing',
        pretty_name='DrivelologyNarrativeWriting',
        tags=[Tags.KNOWLEDGE, Tags.REASONING],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/drivel-hub',
        subset_list=['narrative-writing-english'],
        metric_list=[{
            'bert_score': {
                'model_id_or_path': 'AI-ModelScope/roberta-large',
                'model_type': 'roberta-large'
            }
        }, {
            'judge_score': {}
        }],
        primary_metric='judge_score',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=NARRATIVE_GENERATION_TEMPLATE
    )
)
class DrivelologyNarrativeWritingAdapter(DefaultDataAdapter):

    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_batch_scoring = True  # Enable batch scoring

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.
        """
        text = record['text']
        reference_narrative = record['narrative']

        # Format the generation prompt with the text
        input_prompt = NARRATIVE_GENERATION_TEMPLATE.format(text=text)

        # Create content list for the input
        content_list = [ContentText(text=input_prompt)]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=reference_narrative,
            metadata={
                'text': text,
                'reference_narrative': reference_narrative
            }
        )

    def batch_match_score(self, original_predictions, filtered_predictions, references, task_states):
        """
        Batch calculate the match scores using BERTScore.
        """
        from evalscope.metrics.nlp.metrics import BertScore

        score_args = self.get_metric_args('bert_score')
        bert_scorer = BertScore(**score_args)
        bert_score_f1 = bert_scorer.apply(filtered_predictions, references)
        scores = []
        for i in range(len(original_predictions)):
            score = Score(
                extracted_prediction=filtered_predictions[i],
                prediction=original_predictions[i],
                value={'bert_score': bert_score_f1[i]}
            )
            scores.append(score)
        return scores

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        return [JudgeCase(case_id='rating', output_contract=RATING_CONTRACT)]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        prompt = NARRATIVE_EVALUATION_TEMPLATE.format(
            candidate=context.filtered_prediction,
            reference=context.reference,
        )
        prompt += case.output_contract.instruction()
        return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        rating = case_verdicts[0].value.rating
        # The official metric is the 1-5 rating normalised onto [0, 1].
        return ReducedVerdict(value={'judge_score': (rating - 1) / 4.0}, metadata={'rating': rating})

    def finalize_judge_score(self, review, context) -> Score:
        score = super().finalize_judge_score(review, context)
        score.main_score_name = 'judge_score'
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Aggregate scores across all samples.

        Each metric is averaged only over the samples that actually carry it: a sample whose judge
        review was unusable has no ``judge_score`` and is excluded from that mean rather than
        counted as 0, while its rule-based ``bert_score`` still contributes.
        """
        results: List[AggScore] = []
        for metric_name in ('judge_score', 'bert_score'):
            values = [ss.score.value[metric_name] for ss in sample_scores if metric_name in ss.score.value]
            if not values:
                results.append(AggScore(metric_name=metric_name, score=0.0, num=0, metadata={}))
                continue
            results.append(
                AggScore(
                    metric_name=metric_name,
                    score=sum(values) / len(values),
                    num=len(values),
                    metadata={
                        'min_score': min(values),
                        'max_score': max(values),
                    },
                )
            )
        return results
