from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)"."""


@register_benchmark(
    BenchmarkMeta(
        name='docmath',
        pretty_name='DocMath',
        tags=[Tags.REASONING, Tags.MATH, Tags.LONG_CONTEXT],
        description=
        'DocMath-Eval is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires the model to comprehend long and specialized documents and perform numerical reasoning to answer the given question.',  # noqa: E501
        dataset_id='yale-nlp/DocMath-Eval',
        metric_list=['acc'],
        subset_list=['complong_testmini', 'compshort_testmini', 'simplong_testmini', 'simpshort_testmini'],
        eval_split='test',
        prompt_template=TEMPLATE_0SHOT,
    )
)
class DocMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_llm_judge = True  # Enable LLM judge for DocMath
        self.split_as_subset = True  # Use split as subset for DocMath

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        ground_truth = record['ground_truth']

        return Sample(
            input=record['question'],
            target=str(ground_truth),
            metadata={
                'question_id': record.get('question_id', ''),
                'paragraphs': record['paragraphs'],
                'answer_type': type(ground_truth).__name__
            }
        )

    def format_prompt_template(self, sample):
        context = '\n'.join(sample.metadata['paragraphs'])
        question = sample.input
        return self.prompt_template.format(context=context, question=question)

    def extract_answer(self, prediction: str, task_state: TaskState):
        """
        Extract the answer from the model prediction.
        """
        from .utils import extract_answer

        extracted_answer = extract_answer(prediction)
        return extracted_answer

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """
        Calculate accuracy score by matching prediction with reference.
        """
        from .utils import get_acc

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        answer_type = task_state.metadata.get('answer_type', 'unknown')
        accuracy = get_acc(prediction=filtered_prediction, gt=reference, answer_type=answer_type)
        score.value = {'acc': accuracy}
        score.main_score_name = 'acc'

        return score

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """
        Use LLM judge to evaluate the prediction against the reference.
        """
        from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        question = task_state.metadata.get('question', '')

        # Get grading response
        prompt = ORM_USER_TEMPLATE.format(problem=question, answer_1=reference, answer_2=filtered_prediction)
        orm_response = self.llm_judge.judge(prompt, system_prompt=GENERAL_ORM_PROMPT)

        # Parse grading response
        if 'YES' in orm_response:
            accuracy = 1.0
        else:
            accuracy = 0.0

        score.value = {'acc': accuracy}
        score.explanation = f'LLM judge: {orm_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id
        }
        score.main_score_name = 'acc'

        return score
