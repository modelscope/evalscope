from pydantic import BaseModel
from typing import Any, Dict, List, Literal

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


# The judge prompt requires the verdict as "[[YES]]" or "[[NO]]" after its explanation; a bare
# "YES" anywhere in that explanation must not decide the score.
class Equivalence(BaseModel):
    reasoning: str = ''
    verdict: Literal['YES', 'NO']


EQUIVALENCE_CONTRACT = OutputContract(schema_model=Equivalence)

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
        description="""
## Overview

DocMath-Eval is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires models to comprehend long and specialized documents and perform numerical reasoning to answer questions.

## Task Description

- **Task Type**: Document-based Mathematical Reasoning
- **Input**: Long document context + numerical reasoning question
- **Output**: Numerical answer with reasoning
- **Focus**: Long-context comprehension and quantitative reasoning

## Key Features

- Long specialized documents requiring comprehension
- Numerical reasoning within document context
- Multiple complexity levels (comp/simp, long/short)
- Tests real-world document understanding
- Requires both reading comprehension and math skills

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM-as-judge for answer evaluation
- Subsets: complong_testmini, compshort_testmini, simplong_testmini, simpshort_testmini
- Answer format: "Therefore, the answer is (answer)"
""",  # noqa: E501
        dataset_id='yale-nlp/DocMath-Eval',
        metric_list=['acc'],
        subset_list=['complong_testmini', 'compshort_testmini', 'simplong_testmini', 'simpshort_testmini'],
        eval_split='test',
        prompt_template=TEMPLATE_0SHOT,
    )
)
class DocMathAdapter(DefaultDataAdapter):

    scoring_policy = ScoringPolicy.JUDGE_DEFAULT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        context = '\n'.join(record['paragraphs'])
        question = record['question']
        message = self.prompt_template.format(context=context, question=question)
        return Sample(
            input=[ChatMessageUser(content=message)],
            target=str(ground_truth),
            metadata={
                'question_id': record.get('question_id', ''),
                'answer_type': type(ground_truth).__name__
            }
        )

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

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE
            prompt = ORM_USER_TEMPLATE.format(
                problem=judge_context.task_state.metadata.get('question', ''),
                answer_1=judge_context.reference,
                answer_2=judge_context.filtered_prediction,
            ) + case.output_contract.instruction()
            return JudgeRequest(
                messages=[ChatMessageSystem(content=GENERAL_ORM_PROMPT),
                          ChatMessageUser(content=prompt)]
            )

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            return ReducedVerdict(value={'acc': 1.0 if case_verdicts[0].value.verdict == 'YES' else 0.0})

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='equivalence', output_contract=EQUIVALENCE_CONTRACT)],
            request=request,
            reduce=reduce,
            main_score_name='acc'
        )
