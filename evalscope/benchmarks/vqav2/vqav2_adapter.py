from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.caption.base import VQACaptionAdapter
from evalscope.constants import HubType, Tags

PROMPT_TEMPLATE = """Answer the question according to the image using a short phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes)."""


@register_benchmark(
    BenchmarkMeta(
        name='vqav2',
        pretty_name='VQAv2',
        description="""
## Overview

VQAv2 is the balanced Visual Question Answering benchmark built on COCO images. It evaluates whether
multimodal models can answer open-ended natural-language questions grounded in image content.

## Task Description

- **Task Type**: Open-ended visual question answering
- **Input**: Image + natural-language question
- **Output**: Short answer phrase
- **Domains**: General image understanding, object recognition, counting, attributes, relations

## Evaluation Notes

- Default data source: `lmms-lab/VQAv2` on ModelScope, `validation` split
- Hugging Face remains available by setting `extra_params.dataset_hub="huggingface"`
- Primary metric: **VQAv2 soft accuracy** over human annotator answers
- Also reports normalized exact match against the available answer set
- The adapter accepts common answer formats: list of strings, list of answer dicts, or `multiple_choice_answer`
""",
        tags=[Tags.MULTI_MODAL, Tags.QA],
        dataset_id='lmms-lab/VQAv2',
        paper_url='https://arxiv.org/abs/1612.00837',
        subset_list=['default'],
        metric_list=['vqa_score', 'exact_match'],
        eval_split='validation',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'dataset_hub': {
                'type': 'str',
                'description': 'Dataset hub used to load VQAv2 annotations and images.',
                'value': HubType.MODELSCOPE,
                'choices': [HubType.HUGGINGFACE, HubType.MODELSCOPE, HubType.LOCAL],
            },
            'eval_split': {
                'type': 'str',
                'description': 'Source split to load; defaults to validation.',
                'value': '',
            },
            'dataset_revision': {
                'type': 'str',
                'description': 'Optional dataset revision; leave empty to use the hub default.',
                'value': '',
            },
            'image_dir': {
                'type': 'str',
                'description': 'Optional local directory containing VQAv2 images for local JSONL/CSV data.',
                'value': '',
            },
            'image_extension': {
                'type': 'str',
                'description': 'Optional extension override for local images, for example "jpg".',
                'value': '',
            },
        },
    )
)
class VQAv2Adapter(VQACaptionAdapter):
    """Adapter for VQAv2 open-ended visual question answering."""

    source_dataset_ids = {
        HubType.MODELSCOPE: 'lmms-lab/VQAv2',
        HubType.HUGGINGFACE: 'lmms-lab/VQAv2',
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
