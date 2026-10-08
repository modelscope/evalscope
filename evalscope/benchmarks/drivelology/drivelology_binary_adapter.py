# flake8: noqa: E501

from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentText
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.yes_no import extract_verdict

DESCRIPTION = """
## Overview

Drivelology Binary Classification evaluates models' ability to identify "drivelology" - a unique linguistic phenomenon characterized as "nonsense with depth." These are utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.

## Task Description

- **Task Type**: Binary Text Classification (Yes/No)
- **Input**: Text sample to classify
- **Output**: "Yes" if drivelology, "No" otherwise
- **Domain**: Linguistic analysis, humor detection, pragmatics

## Key Features

- Tests understanding of layered linguistic meanings
- Distinguishes nonsense-with-depth from pure nonsense and normal text
- Requires contextual understanding and emotional insight
- Covers humor, irony, sarcasm detection
- Multiple difficulty levels available

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score
- Subsets: binary-english-easy, binary-english-hard, binary-chinese-easy, binary-chinese-hard
"""

PROMPT_TEMPLATE = """
#Instruction#:
Classify whether the given text is a Drivelology sample or not.

#Definition#:
- Drivelology: Statements that appear logically coherent but contain deeper, often paradoxical meanings.
These challenge conventional interpretation by blending surface-level nonsense with underlying depth,
often incorporating elements of humor, irony, or sarcasm, and requiring contextual understanding and
emotional insight to unravel their true significance.
- non-Drivelology: This includes pure nonsense (grammatically correct but semantically meaningless
statements, such as "Colourless green ideas sleep furiously") and normal sentences, including quotes
or proverbs, that convey clear or straightforward information without the layered complexity
characteristic of Drivelology.

#Output Format#:
You should try your best to answer "Yes" if the given input text is Drivelology, otherwise specify "No".
The answer you give MUST be \"Yes\" or \"No\"".

#Input Text#: {text}
#Your Answer#:
""".strip()  # noqa: E501

FEWSHOT_PROMPT_TEMPLATE = """
#Instruction#:
Classify whether the given text is a Drivelology sample or not.

#Definition#:
- Drivelology: Statements that appear logically coherent but contain deeper, often paradoxical meanings.
These challenge conventional interpretation by blending surface-level nonsense with underlying depth,
often incorporating elements of humor, irony, or sarcasm, and requiring contextual understanding and
emotional insight to unravel their true significance.
- non-Drivelology: This includes pure nonsense (grammatically correct but semantically meaningless
statements, such as "Colourless green ideas sleep furiously") and normal sentences, including quotes
or proverbs, that convey clear or straightforward information without the layered complexity
characteristic of Drivelology.

#Output Format#:
You should try your best to answer "Yes" if the given input text is Drivelology, otherwise specify "No".
The answer you give MUST be \"Yes\" or \"No\"".

Here are some examples of how to solve similar problems:

#Input Text#: Saw a book called "how to solve 50 percent of your problems" so I bought 2 books.
#Your Answer#: Yes

#Input Text#: Colourless green ideas sleep furiously.
#Your Answer#: No

#Input Text#: I went to a restaurant, and saw this guy was choking. I gotta save him. And then I realized he was just speaking French.
#Your Answer#: Yes

#Input Text#: Either it is or it isn't.
#Your Answer#: No

#Input Text#: {text}
#Your Answer#:
""".strip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='drivel_binary',
        pretty_name='DrivelologyBinaryClassification',
        tags=[Tags.YES_NO],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/drivel-hub',
        subset_list=['binary-classification'],
        metric_list=['accuracy', 'precision', 'recall', 'f1_score', 'yes_ratio'],
        primary_metric='accuracy',
        aggregation='f1',
        few_shot_num=0,
        few_shot_mode='fixed',
        allowed_few_shot_nums=(0, 4),
        eval_split='test',
        prompt_template='{question}',
        few_shot_prompt_template='{question}',
        evaluation_version='v1.1',
    )
)
class DrivelologyBinaryClassificationAdapter(DefaultDataAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_overall_metric = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        if self.few_shot_num > 0:
            prompt = FEWSHOT_PROMPT_TEMPLATE.format(text=record['text'])
        else:
            prompt = PROMPT_TEMPLATE.format(text=record['text'])
        content_list: List[Content] = [ContentText(text=prompt)]
        answer = 'YES' if str(record['label']) == 'drivelology' else 'NO'  # 'YES' or 'NO'
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=answer,
            metadata={
                'answer': answer,
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        verdict = extract_verdict(filtered_prediction, allow_lowercase_exact=True)
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        # Credit only an exact whole-word verdict match; both verdicts or neither scores 0
        score.value = {'acc': 1 if verdict == reference.strip().upper() else 0}
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Custom aggregation to compute accuracy, precision, recall, f1_score, and yes_ratio.
        """

        def compute_metrics(scores: List[SampleScore]):
            tp = fp = fn = 0
            yes_count = 0
            total_count = len(scores)

            for ss in scores:
                gt = ss.sample_metadata['answer'].strip().upper()
                pred = extract_verdict(ss.score.extracted_prediction or '', allow_lowercase_exact=True)
                if pred == 'YES':
                    yes_count += 1
                if pred == 'YES' and gt == 'YES':
                    tp += 1
                elif pred == 'YES' and gt == 'NO':
                    fp += 1
                elif gt == 'YES':
                    fn += 1

            accuracy = sum(ss.score.main_value for ss in scores) / total_count if total_count > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            yes_ratio = yes_count / total_count if total_count > 0 else 0.0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'yes_ratio': yes_ratio,
            }

        overall_metrics = compute_metrics(sample_scores)
        agg_scores = []
        for metric_name, value in overall_metrics.items():
            agg_scores.append(AggScore(metric_name=metric_name, score=value, num=len(sample_scores), metadata={}))

        return agg_scores
