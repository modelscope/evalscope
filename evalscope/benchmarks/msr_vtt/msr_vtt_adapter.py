from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.caption.base import CAPTION_METRICS, DEFAULT_CAPTION_PROMPT, CaptionDatasetAdapter
from evalscope.constants import HubType, Tags


@register_benchmark(
    BenchmarkMeta(
        name='msr_vtt',
        pretty_name='MSR-VTT',
        description="""
## Overview

MSR-VTT is a large-scale open-domain video captioning benchmark for evaluating video-to-text generation.
The native adapter groups records by `video_id`, so multiple annotation rows for one video become one sample
with multiple reference captions.

## Task Description

- **Task Type**: Video captioning
- **Input**: Video clip or URL
- **Output**: One concise natural-language caption
- **Domains**: Open-domain video understanding and description

## Evaluation Notes

- Default data source: `AI-ModelScope/msr-vtt` on ModelScope, `validation` split
- Hugging Face `VLM2Vec/MSR-VTT` remains available by setting `extra_params.dataset_hub="huggingface"`
- Primary metric: **CIDEr**
- Additional metrics: BLEU-1/2/3/4, METEOR, ROUGE-L
- Set `extra_params.video_dir` to prefer local media files over URL metadata
""",
        tags=[Tags.MULTI_MODAL, Tags.IMAGE_CAPTIONING],
        dataset_id='AI-ModelScope/msr-vtt',
        paper_url=
        'https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/',
        subset_list=['default'],
        metric_list=CAPTION_METRICS,
        eval_split='validation',
        prompt_template=DEFAULT_CAPTION_PROMPT,
        extra_params={
            'dataset_hub': {
                'type': 'str',
                'description': 'Dataset hub used to load MSR-VTT annotations.',
                'value': HubType.MODELSCOPE,
                'choices': [HubType.HUGGINGFACE, HubType.MODELSCOPE, HubType.LOCAL],
            },
            'eval_split': {
                'type': 'str',
                'description': 'Source split to load; defaults to validation for ModelScope and test for Hugging Face.',
                'value': '',
            },
            'dataset_revision': {
                'type': 'str',
                'description': 'Optional dataset revision; leave empty to use the hub default.',
                'value': '',
            },
            'video_dir': {
                'type': 'str',
                'description': 'Optional local directory containing MSR-VTT video files.',
                'value': '',
            },
            'video_extension': {
                'type': 'str',
                'description': 'Optional extension override for local videos, for example "mp4".',
                'value': '',
            },
        },
    )
)
class MSRVTTAdapter(CaptionDatasetAdapter):
    """Adapter for MSR-VTT video captioning."""

    media_fields = ['video', 'video_path', 'url']
    reference_fields = ['references', 'caption', 'captions']
    group_key_field = 'video_id'
    source_dataset_ids = {
        HubType.MODELSCOPE: 'AI-ModelScope/msr-vtt',
        HubType.HUGGINGFACE: 'VLM2Vec/MSR-VTT',
    }
    source_eval_splits = {
        HubType.MODELSCOPE: 'validation',
        HubType.HUGGINGFACE: 'test',
    }
    source_subset_names = {
        HubType.MODELSCOPE: {
            'default': None,
        },
        HubType.HUGGINGFACE: {
            'default': 'test_1k',
        },
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
