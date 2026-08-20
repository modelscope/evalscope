from pydantic import BaseModel
from typing import Any, Dict, List, Literal

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample, load_local_file_dataset, resolve_snapshot_or_local_path
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags


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
        name='frames',
        pretty_name='FRAMES',
        tags=[Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems. It evaluates factuality, retrieval accuracy, and reasoning abilities in long-context scenarios.

## Task Description

- **Task Type**: RAG Evaluation / Long-Context QA
- **Input**: Wikipedia context documents + question
- **Output**: Factual answer in specified format
- **Domains**: Factuality, retrieval, multi-hop reasoning

## Key Features

- Tests core RAG capabilities: factuality, retrieval, reasoning
- Provides Wikipedia-sourced context documents
- Questions require synthesizing information from multiple sources
- Evaluates both retrieval quality and answer generation
- Supports both exact match and LLM judge evaluation

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with both exact match and LLM judge
- Response format: "Therefore, the answer is (answer here)"
- Uses normalized answer comparison for exact matching
- LLM judge provides flexible semantic matching
""",  # noqa: E501
        dataset_id='iic/frames',
        metric_list=['acc'],
        eval_split='test',
        prompt_template=TEMPLATE_0SHOT,
    )
)
class FramesAdapter(DefaultDataAdapter):

    scoring_policy = ScoringPolicy.JUDGE_DEFAULT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        dataset_path = resolve_snapshot_or_local_path(self, allow_file_pattern='test.jsonl')
        dataset = load_local_file_dataset(
            adapter=self,
            dataset_path=dataset_path,
            subset='test',
            split=self.eval_split,
            sample_fields=self.record_to_sample,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        )

        test_dataset = DatasetDict({'test': dataset})

        return test_dataset, None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        context = '\n'.join([f"{i['title']}\n{i['text']}" for i in record['wiki_items']])
        question = record['Prompt']

        return Sample(
            input=question, target=record['Answer'], metadata={
                'context': context,
                'wiki_items': record['wiki_items']
            }
        )

    def format_prompt_template(self, sample):
        context = sample.metadata['context']
        question = sample.input
        return self.prompt_template.format(context=context, question=question)

    def extract_answer(self, prediction: str, task_state: TaskState):
        """
        Extract the answer from the model prediction.
        """
        response = prediction.replace('*', '')

        if 'the answer is' in response:
            ans = response.rsplit('the answer is', 1)[-1].strip().strip('.').strip()
        else:
            ans = ''

        return ans

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
        from evalscope.metrics import exact_match
        from .utils import normalize_answer

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        gold = normalize_answer(reference)
        pred = normalize_answer(filtered_prediction)
        accuracy = exact_match(gold=gold, pred=pred)

        score.value = {'acc': accuracy}
        score.main_score_name = 'acc'

        return score

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        return [JudgeCase(case_id='equivalence', output_contract=EQUIVALENCE_CONTRACT)]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE

        prompt = ORM_USER_TEMPLATE.format(
            problem=context.task_state.input_text,
            answer_1=context.reference,
            answer_2=context.filtered_prediction,
        )
        prompt += case.output_contract.instruction()
        return JudgeRequest(messages=[ChatMessageSystem(content=GENERAL_ORM_PROMPT), ChatMessageUser(content=prompt)])

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        return ReducedVerdict(value={'acc': 1.0 if case_verdicts[0].value.verdict == 'YES' else 0.0})

    def finalize_judge_score(self, review, context) -> Score:
        score = super().finalize_judge_score(review, context)
        score.main_score_name = 'acc'
        return score
