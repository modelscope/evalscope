# flake8: noqa: E501

from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentText
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

DESCRIPTION = (
    'Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - '
    'utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, '
    'or rhetorically subversive.'
)

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

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='drivel_binary',
        pretty_name='DrivelologyBinaryClassification',
        tags=[Tags.YES_NO],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/drivel-hub',
        subset_list=['binary-classification'],
        metric_list=['accuracy', 'precision', 'recall', 'f1_score', 'yes_ratio'],
        aggregation='f1',
        few_shot_num=0,
        eval_split='test',
        prompt_template='{question}',
        few_shot_prompt_template='{question}'
    )
)
class DrivelologyBinaryClassificationAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_overall_metric = False
        if self.few_shot_num not in [0, 4]:
            logger.warning(f'For DrivelologyBinaryClassification, use 4-shot by default.')
            self.few_shot_num = 4

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        if self.few_shot_num > 0:
            prompt = FEWSHOT_PROMPT_TEMPLATE.format(text=record['text'])
        else:
            prompt = PROMPT_TEMPLATE.format(text=record['text'])
        content_list: List[Content] = [ContentText(text=prompt)]
        answer = 'YES' if str(record['label']) == 'drivelology' else 'NO'  # 'YES' or 'NO'
        return Sample(input=[ChatMessageUser(content=content_list)], target=answer, metadata={
            'answer': answer,
        })

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        # Check if the reference answer is in the filtered prediction
        result = 1 if reference in filtered_prediction.strip().upper() else 0
        score.value = {'acc': result}
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Custom aggregation to compute accuracy, precision, recall, f1_score, and yes_ratio.
        """

        def compute_metrics(scores: List[SampleScore]):
            tp = fp = tn = fn = 0
            yes_count = 0
            total_count = len(scores)

            for ss in scores:
                gt = ss.sample_metadata['answer'].strip().upper()
                # Get prediction based on score
                pred = gt if ss.score.main_value == 1 else ('NO' if gt == 'YES' else 'YES')
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

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'yes_ratio': yes_ratio
            }

        overall_metrics = compute_metrics(sample_scores)
        agg_scores = []
        for metric_name, value in overall_metrics.items():
            agg_scores.append(AggScore(metric_name=metric_name, score=value, num=len(sample_scores), metadata={}))

        return agg_scores
