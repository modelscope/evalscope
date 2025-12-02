# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.models.utils.openai import chat_messages_from_openai
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='general_vqa',
        pretty_name='General-VQA',
        description='A general visual question answering dataset for custom multimodal evaluation. '
        'Supports OpenAI-compatible message format with images (local paths or base64). '
        'For detailed instructions, please refer to the [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html).',  # noqa: E501
        tags=[Tags.QA, Tags.CUSTOM, Tags.MULTI_MODAL],
        dataset_id='general_vqa',
        metric_list=['BLEU', 'Rouge'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=None,
    )
)
class GeneralVQAAdapter(VisionLanguageAdapter):
    """
    General VQA Adapter for custom multimodal evaluation.

    This adapter supports OpenAI-compatible message format for multimodal inputs:
    - TSV format: Tab-separated file with 'messages' and 'answer' columns
    - JSONL format: JSON lines with 'messages' and 'answer' fields

    The 'messages' field should be a JSON string (in TSV) or object (in JSONL)
    following OpenAI chat completion format with support for:
    - Text content: {"type": "text", "text": "..."}
    - Image URLs: {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}
    - Base64 images: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record with 'messages' and 'answer' fields.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        messages_data = record.get('messages')
        answer = record.get('answer', '')

        # Parse messages if it's a JSON string (from TSV)
        if isinstance(messages_data, str):
            try:
                messages_data = json.loads(messages_data)
            except json.JSONDecodeError as e:
                logger.error(f'Failed to parse messages JSON: {e}')
                raise

        # Convert messages to ChatMessage objects using the standard OpenAI parser
        if isinstance(messages_data, list):
            message_list = chat_messages_from_openai(model='', messages=messages_data)
        else:
            logger.warning(f'Unexpected messages format: {type(messages_data)}')
            message_list = []

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
