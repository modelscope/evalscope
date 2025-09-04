import os
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, LocalDataLoader, Sample
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
        name='frames',
        pretty_name='FRAMES',
        tags=[Tags.REASONING, Tags.LONG_CONTEXT],
        description=
        'FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.',  # noqa: E501
        dataset_id='iic/frames',
        metric_list=['acc'],
        eval_split='test',
        prompt_template=TEMPLATE_0SHOT,
    )
)
class FramesAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_llm_judge = True  # Enable LLM judge for FRAMES

    def load(self):
        # Try to load dataset from local disk
        dataset_name_or_path = self.dataset_id
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from {dataset_name_or_path}')
            dataset_path = dataset_name_or_path
        else:
            from modelscope import dataset_snapshot_download

            # Load dataset from remote
            logger.info(f'Loading dataset from modelscope: > dataset_name: {dataset_name_or_path}')
            # download dataset snapshot
            dataset_path = dataset_snapshot_download(dataset_name_or_path, allow_file_pattern='test.jsonl')

        dataset = LocalDataLoader(
            data_id_or_path=dataset_path,
            split=self.eval_split,
            sample_fields=self.record_to_sample,
            subset='test',
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        ).load()

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

        question = task_state.input_text

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
