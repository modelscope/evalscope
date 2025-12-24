# flake8: noqa: E501
import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.metric.scorer import AggScore, SampleScore
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

CRITIQUE_TEMPLATE = """The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \boxed{{}}.
"""


@register_benchmark(
    BenchmarkMeta(
        name='process_bench',
        pretty_name='ProcessBench',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'ProcessBench is a benchmark for evaluating AI models on mathematical reasoning tasks. It includes various subsets such as GSM8K, Math, OlympiadBench, and OmniMath, each with its own set of problems that require step-by-step reasoning to arrive at the correct answer.',  # noqa: E501
        dataset_id='Qwen/ProcessBench',
        subset_list=['gsm8k', 'math', 'olympiadbench', 'omnimath'],
        metric_list=['error_acc', 'correct_acc', 'simple_f1_score'],
        aggregation='f1',
        eval_split='test',
        prompt_template=CRITIQUE_TEMPLATE
    )
)
class ProcessBenchAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_as_subset = True  # Use split as subset

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        problem = record['problem']
        steps = record['steps']
        tagged_response = ''
        for sdx, step in enumerate(steps):
            tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
        tagged_response = tagged_response.strip()

        return Sample(
            input=problem,
            target=str(record['label']),
            metadata={
                'steps': steps,
                'tagged_response': tagged_response,
                'final_answer_correct': record['final_answer_correct']
            }
        )

    def format_prompt_template(self, sample):
        """Format the prompt template with problem and tagged response."""
        problem = sample.input
        tagged_response = sample.metadata['tagged_response']
        return self.prompt_template.format(problem=problem, tagged_response=tagged_response)

    def extract_answer(self, prediction: str, task_state: TaskState):
        """Extract the answer from the model prediction."""
        pred = self._extract_answer_from_text(prediction)
        try:
            pred = int(pred)
        except Exception:
            pred = None
        return pred

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Calculate evaluation scores by comparing prediction with reference."""
        score = Score(
            extracted_prediction=str(filtered_prediction) if filtered_prediction is not None else None,
            prediction=original_prediction,
        )

        # Convert filtered_prediction to int if possible
        try:
            pred_int = int(filtered_prediction) if filtered_prediction is not None else None
        except (ValueError, TypeError):
            pred_int = None

        # Calculate accuracy
        reference = int(reference) if reference is not None else None
        accuracy = 1.0 if reference == pred_int else 0.0

        # Determine metric name based on label
        if reference == -1:
            metric_name = 'correct_acc'
        else:
            metric_name = 'error_acc'

        score.value = {metric_name: accuracy}
        score.main_score_name = metric_name

        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate scores to compute final metrics including F1 score."""
        correct_scores = []
        error_scores = []

        for sample_score in sample_scores:
            score = sample_score.score
            if 'correct_acc' in score.value:
                correct_scores.append(score.value['correct_acc'])
            elif 'error_acc' in score.value:
                error_scores.append(score.value['error_acc'])

        agg_list = []

        if correct_scores:
            agg_list.append(
                AggScore(
                    metric_name='correct_acc', score=sum(correct_scores) / len(correct_scores), num=len(correct_scores)
                )
            )

        if error_scores:
            agg_list.append(
                AggScore(metric_name='error_acc', score=sum(error_scores) / len(error_scores), num=len(error_scores))
            )

        # Calculate simple F1 score
        if correct_scores and error_scores:
            from evalscope.metrics import simple_f1_score
            agg_list.append(
                AggScore(
                    metric_name='simple_f1_score',
                    score=simple_f1_score((correct_scores, error_scores)),
                    num=len(correct_scores) + len(error_scores)
                )
            )

        return agg_list

    @staticmethod
    def _extract_answer_from_text(solution_text: str):
        """Extract answer from solution text using boxed pattern."""
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(boxed_pattern, solution_text)
        if matches:
            return matches[-1].strip()
        return None
