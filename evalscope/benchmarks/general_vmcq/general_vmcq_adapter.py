# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessage
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.models.utils.openai import chat_messages_from_openai
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='general_vmcq',
        pretty_name='General-VMCQ',
        description='A general visual multiple-choice question answering dataset for custom multimodal evaluation. '
        'Supports OpenAI-compatible message format with images (local paths or base64). '
        'For detailed instructions, please refer to the [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html#general-vmcq).',  # noqa: E501
        tags=[Tags.MULTIPLE_CHOICE, Tags.CUSTOM, Tags.MULTI_MODAL],
        dataset_id='general_vmcq',
        subset_list=['default'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='dev',
        eval_split='val',
        prompt_template=None,
    )
)
class GeneralVMCQAdapter(MultiChoiceAdapter):
    """
    General VMCQ (Visual Multiple Choice Question) Adapter for custom multimodal evaluation.
    
    This adapter supports OpenAI-compatible message format for multimodal inputs:
    - TSV format: Tab-separated file with 'messages' and 'answer' columns
    - JSONL format: JSON lines with 'messages' and 'answer' fields
    
    The 'messages' field should be a JSON string (in TSV) or object (in JSONL)
    following OpenAI chat completion format with support for:
    - Text content: {"type": "text", "text": "..."}
    - Image URLs: {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}
    - Base64 images: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    
    The messages should contain the question text with choices already formatted (e.g., "A. Option1\\nB. Option2").
    The 'answer' field should contain the correct choice letter(s) (e.g., "A", "B", "AB").
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

        # For MCQ, we return the messages as input (already formatted with choices)
        # The choices are embedded in the question text, not as a separate field
        return Sample(
            input=message_list,
            target=answer or '',
            metadata={'id': record.get('id', 'unknown')},
        )
