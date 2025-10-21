# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser, dict_to_chat_message
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = '请回答问题\n{question}'


@register_benchmark(
    BenchmarkMeta(
        name='general_qa',
        pretty_name='General-QA',
        description='A general question answering dataset for custom evaluation. '
        'For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#qa).',  # noqa: E501
        tags=[Tags.QA, Tags.CUSTOM],
        dataset_id='general_qa',
        metric_list=['BLEU', 'Rouge'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class GeneralQAAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        query = record.get('question') or record.get('query')
        answer = record.get('answer') or record.get('response')
        system_prompt = record.get('system')
        messages = record.get('messages')

        message_list = []
        if messages:
            message_list = [dict_to_chat_message(m) for m in messages]
        else:
            if system_prompt:
                message_list.append(ChatMessageSystem(content=system_prompt))
            message_list.append(ChatMessageUser(content=query))

        return Sample(input=message_list, target=answer or '')

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Calculate scores for each configured metric
        for metric in self.metric_list:
            try:
                if metric == 'Rouge':
                    from evalscope.metrics.rouge_metric import compute_rouge_score_one_sample_zh

                    score.value.update(compute_rouge_score_one_sample_zh([filtered_prediction], [reference]))
                elif metric == 'BLEU':
                    from evalscope.metrics import bleu_ngram_one_sample

                    score.value.update(bleu_ngram_one_sample(filtered_prediction, reference))
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                return None

        score.main_score_name = 'Rouge-L-R'
        return score
