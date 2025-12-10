import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentText
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = (
    'Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - '
    'utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, '
    'or rhetorically subversive.'
)

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

Please format your rating strictly as: "Rating: [[X]]" where X is a whole number from 1 to 5.

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
            'gpt_score': {}
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=NARRATIVE_GENERATION_TEMPLATE
    )
)
class DrivelologyNarrativeWritingAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_llm_judge = True  # Use LLM as a judge by default
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
        from evalscope.metrics.metric import BertScore

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

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """
        Calculate the match score using LLM judge and BERTScore.
        """
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Initialize score value dictionary
        score.value = {}

        # Use LLM judge to evaluate narrative quality
        eval_prompt = NARRATIVE_EVALUATION_TEMPLATE.format(candidate=filtered_prediction, reference=reference)

        judge_response = self.llm_judge.judge(eval_prompt)
        logger.info(f'LLM judge response received (first 100 chars): {judge_response[:100]}...')

        # Extract rating using regex pattern
        match = re.search(r'Rating:\s*\[\[([1-5])\]\]', judge_response)
        if match:
            rating = int(match.group(1))
            gpt_score = (rating - 1) / 4.0  # Normalize to 0-1 scale
            logger.info(f'Rating extracted: {rating}/5 -> {gpt_score}')
        else:
            # Try alternative pattern
            alt_match = re.search(r'(\[\[|\[)([1-5])(\]\]|\])', judge_response)
            if alt_match:
                rating = int(alt_match.group(2))
                gpt_score = (rating - 1) / 4.0
                logger.info(f'Rating extracted (alt pattern): {rating}/5 -> {gpt_score}')
            else:
                # Last resort: standalone digit
                number_match = re.search(r'(?<!\d)[1-5](?!\d)', judge_response)
                if number_match:
                    rating = int(number_match.group(0))
                    gpt_score = (rating - 1) / 4.0
                    logger.info(f'Rating extracted (fallback): {rating}/5 -> {gpt_score}')
                else:
                    gpt_score = 0.0
                    logger.warning('No rating found in response, using default 0.0')

        score.value['gpt_score'] = gpt_score
        score.explanation = f'LLM judge rating: {gpt_score:.2f}'

        score.metadata = {
            'judge_response': judge_response[:300],
            'model': getattr(self.llm_judge, 'model_id', 'unknown')
        }

        score.main_score_name = 'gpt_score'
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Aggregate scores across all samples.
        """
        if not sample_scores:
            return [
                AggScore(metric_name='gpt_score', score=0.0, num=0, metadata={}),
                AggScore(metric_name='bert_score', score=0.0, num=0, metadata={})
            ]

        # Extract scores
        gpt_scores = [ss.score.value.get('gpt_score', 0.0) for ss in sample_scores]
        bert_scores = [ss.score.value.get('bert_score', 0.0) for ss in sample_scores]

        # Calculate averages
        avg_gpt_score = sum(gpt_scores) / len(gpt_scores) if gpt_scores else 0.0
        avg_bert_score = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0

        return [
            AggScore(
                metric_name='gpt_score',
                score=avg_gpt_score,
                num=len(sample_scores),
                metadata={
                    'min_score': min(gpt_scores),
                    'max_score': max(gpt_scores)
                }
            ),
            AggScore(
                metric_name='bert_score',
                score=avg_bert_score,
                num=len(sample_scores),
                metadata={
                    'min_score': min(bert_scores),
                    'max_score': max(bert_scores)
                }
            )
        ]
