from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentText
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = """
## Overview

CoinFlip is a symbolic reasoning benchmark that tests LLMs' ability to track binary state changes through sequences of actions. Each problem involves determining a coin's final state (heads/tails) after various flipping operations.

## Task Description

- **Task Type**: Symbolic Reasoning / State Tracking
- **Input**: Description of coin flip operations by different people
- **Output**: Final coin state (YES for heads-up, NO for tails-up)
- **Focus**: Binary state tracking and logical inference

## Key Features

- Tests state tracking through action sequences
- Binary reasoning (flip/no-flip) decisions
- Requires careful attention to operator effects
- Evaluates systematic logical reasoning
- Clear, unambiguous answers

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should follow "ANSWER: YES/NO" format
- Five metrics: accuracy, precision, recall, F1, yes_ratio
- F1 score is the primary aggregation metric
- Supports few-shot evaluation with reasoning examples
"""

PROMPT_TEMPLATE = """
Solve the following coin flip problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer YES or NO to the problem.

Reasoning:
"""  # noqa: E501

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}

""".lstrip() + PROMPT_TEMPLATE  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='coin_flip',
        pretty_name='CoinFlip',
        tags=[Tags.REASONING, Tags.YES_NO],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/coin-flip',
        metric_list=['accuracy', 'precision', 'recall', 'f1_score', 'yes_ratio'],
        aggregation='f1',
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CoinFlipAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_overall_metric = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        answer = record['answer']
        input_text = self.prompt_template.format(question=question)
        content_list: List[Content] = [ContentText(text=input_text)]
        answer = str(answer).upper()  # 'YES' or 'NO'
        return Sample(input=[ChatMessageUser(content=content_list)], target=answer, metadata={
            'answer': answer,
        })

    def extract_answer(self, prediction, task_state):
        import re

        match = re.search(r'ANSWER:\s*(.*)', prediction)
        return match.group(1) if match else prediction

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        # Check for an exact match against the extracted answer.
        result = 1 if reference in filtered_prediction else 0
        score.value = {'acc': result}
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Custom aggregation to compute accuracy, precision, recall, f1_score, and yes_ratio.
        """

        tp = fp = tn = fn = 0
        yes_count = 0
        total_count = len(sample_scores)

        for ss in sample_scores:
            gt = ss.sample_metadata['answer'].strip().upper()
            pred = ss.score.extracted_prediction.strip().upper()

            if pred == 'YES':
                yes_count += 1
            if pred == 'YES' and gt == 'YES':
                tp += 1
            elif pred == 'YES' and gt == 'NO':
                fp += 1
            elif pred == 'NO' and gt == 'NO':
                tn += 1
            elif pred == 'NO' and gt == 'YES':
                fn += 1

        accuracy = (tp + tn) / total_count if total_count > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        yes_ratio = yes_count / total_count if total_count > 0 else 0.0

        overall_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'yes_ratio': yes_ratio
        }

        agg_scores = []
        for metric_name, value in overall_metrics.items():
            agg_scores.append(AggScore(metric_name=metric_name, score=value, num=len(sample_scores), metadata={}))

        return agg_scores
