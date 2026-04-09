from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.metric import Score
from evalscope.api.registry import get_metric, register_benchmark
from evalscope.constants import Tags, HubType
from evalscope.utils.logger import get_logger

from .utils import strip_string, extract_answer

logger = get_logger()

SUBSET_LIST = [
    "atkins",
    "calculus",
    "chemmc",
    "class",
    "diff",
    "fund",
    "matter",
    "quan",
    "stat",
    "thermo"
]

QA_TEMPLATE = """Answer the following question. Please reason step by step, and put your final answer within \\boxed{{}}.
Your final answer must be  numeric-only, and it's unit should be {unit}.

{question}
"""

@register_benchmark(
    BenchmarkMeta(
        name='scibench',
        pretty_name='SciBench',
        description='SciBench from <SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models>',
        tags=[Tags.CUSTOM],
        dataset_id='xw27/scibench',
        subset_list=SUBSET_LIST,
        metric_list=['scibench_numeric_acc'],
        few_shot_num=0,
        eval_split='train',
        prompt_template=QA_TEMPLATE
    )
)

class SciBenchAdapter(DefaultDataAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_hub=HubType.HUGGINGFACE
        self.reformat_subset = True

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['problem_text'],
            target=record['answer_number'],
            subset_key=record['source'],
            metadata={
                'answer_latex': record['answer_latex'],
                'solution': record['solution'],
                'comments': record['comment'],
                'unit': record['unit'],
                'id': record['problemid']
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the basic prompt template with the sample data.

        This method applies the prompt template to format the input text
        for models when no few-shot examples are used.

        Args:
            sample (Sample): The sample object containing the prompt data

        Returns:
            str: The formatted prompt ready for model input
        """
        return QA_TEMPLATE.format(
            question=sample.input,
            unit=sample.metadata['unit']
        )
    
    def extract_answer(self, prediction: str, task_state) -> str:
        """
        Hook method for custom answer extraction from model predictions.

        This method can be overridden in subclasses to implement specific
        logic for extracting the final answer from complex model outputs.

        Args:
            prediction (str): The model prediction to extract from
            task_state (TaskState): The task state for additional context

        Returns:
            str: The extracted answer
        """
        return strip_string(
            extract_answer(
                prediction=prediction,
                task_state=task_state,
            )
        )
    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        """
        """
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction
        )

        try:
            metric_scorer = get_metric("scibench_numeric_acc")
            score.explanation = f"scibench_numeric_acc from {filtered_prediction}, {reference}"
            metric_func = metric_scorer()
            metric_score = metric_func(
                prediction=filtered_prediction,
                reference=reference,
            )
            score.explanation += f"\nmetric is {metric_score}"
            score.value['acc'] = metric_score
        except Exception as e:
            # Handle evaluation errors
            score.value['acc'] = 0
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata.update({'error': str(e)})

        return score
