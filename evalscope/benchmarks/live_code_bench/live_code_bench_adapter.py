from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import convert_numpy_types
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench',
        pretty_name='Live-Code-Bench',
        tags=[Tags.CODING],
        description=
        'Live Code Bench is a benchmark for evaluating code generation models on real-world coding tasks. It includes a variety of programming problems with test cases to assess the model\'s ability to generate correct and efficient code solutions.',  # noqa: E501
        dataset_id='AI-ModelScope/code_generation_lite',
        subset_list=['release_latest'],
        metric_list=['Pass@1'],
        eval_split='test',
        prompt_template=
        '### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n',
        extra_params={
            'start_date': None,
            'end_date': None,
            'timeout': 6,
            'debug': False
        },
    )
)
class LiveCodeBenchAdapter(DefaultDataAdapter):
    """
    Live Code Bench adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.timeout = self.extra_params.get('timeout', 6)
        self.debug = self.extra_params.get('debug', False)
        self.start_date = self.extra_params.get('start_date')
        self.end_date = self.extra_params.get('end_date')

        self.save_metadata = False  # Don't save metadata, since they are large

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        from .load_utils import transform

        record = transform(record)

        question_content = record['question_content']
        format_prompt = record['format_prompt']
        full_prompt = self.prompt_template.format(question_content=question_content, format_prompt=format_prompt)

        return Sample(
            input=[ChatMessageUser(content=full_prompt)],
            target='',
            metadata={
                'evaluation_sample': record['evaluation_sample'],
                'contest_date': record['contest_date']
            }
        )

    def sample_filter(self, sample):
        from .load_utils import filter_date

        return filter_date(sample.metadata['contest_date'], start_date=self.start_date, end_date=self.end_date)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""
        from .extract_utils import extract_code_generation
        return extract_code_generation(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from .evaluate_utils import codegen_metrics

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        references = [{'input_output': task_state.metadata['evaluation_sample']}]
        predictions = [[filtered_prediction]]

        try:
            metrics, eval_results, final_metadata = codegen_metrics(
                references,
                predictions,
                k_list=[1],
                num_process_evaluate=1,
                timeout=self.timeout,
                debug=self.debug,
            )
            pass_rate = metrics['pass@1'] / 100  # convert to point scale

            score.value = {'pass': float(pass_rate > 0)}
            score.explanation = f"Pass@1: {metrics['pass@1']}%"

            # Convert numpy types to native Python types for JSON serialization
            serializable_eval_results = convert_numpy_types(eval_results)
            serializable_final_metadata = convert_numpy_types(final_metadata)

            score.metadata = {
                'pass_rate': float(pass_rate),
                'timeout': self.timeout,
                'debug': self.debug,
                'eval_results': serializable_eval_results,
                'final_metadata': serializable_final_metadata
            }
        except Exception as e:
            score.value = {'pass': False}
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata = {'error': str(e)}

        score.main_score_name = 'pass'
        return score

    def aggregate_scores(self, sample_scores):
        from evalscope.metrics.metric import PassAtK

        # calculate pass@k here
        agg_list = []
        for metric in self.metric_list:
            if metric.lower().startswith('pass@'):
                k = int(metric.split('@')[1])
                # Get the scores for this metric
                agg = PassAtK(k)
                agg_list.extend(agg(sample_scores))
        return agg_list
