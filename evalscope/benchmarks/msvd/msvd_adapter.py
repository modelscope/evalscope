from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.caption.base import CAPTION_METRICS, DEFAULT_CAPTION_PROMPT, CaptionDatasetAdapter
from evalscope.constants import HubType, Tags


@register_benchmark(
    BenchmarkMeta(
        name='msvd',
        pretty_name='MSVD',
        description="""
## Overview

MSVD is a classic video captioning benchmark with short web videos annotated by many human captions.
The native adapter treats each video as one evaluation sample and uses all available captions as references.

## Task Description

- **Task Type**: Video captioning
- **Input**: Video clip
- **Output**: One concise natural-language caption
- **Domains**: Open-domain video understanding and description

## Evaluation Notes

- Default data source: `VLM2Vec/MSVD` on Hugging Face, `test` split
- A ModelScope MSVD mirror was not available when this adapter was added; set `dataset_id` or `local_path`
  together with `extra_params.dataset_hub` if a mirror becomes available
- Primary metric: **CIDEr**
- Additional metrics: BLEU-1/2/3/4, METEOR, ROUGE-L
- Set `extra_params.video_dir` when the dataset only provides video file names and local media files are required
""",
        tags=[Tags.MULTI_MODAL, Tags.IMAGE_CAPTIONING],
        dataset_id='VLM2Vec/MSVD',
        paper_url='https://aclanthology.org/P11-1020/',
        subset_list=['default'],
        metric_list=CAPTION_METRICS,
        eval_split='test',
        prompt_template=DEFAULT_CAPTION_PROMPT,
        extra_params={
            'dataset_hub': {
                'type': 'str',
                'description': 'Dataset hub used to load MSVD annotations.',
                'value': HubType.HUGGINGFACE,
                'choices': [HubType.HUGGINGFACE, HubType.MODELSCOPE, HubType.LOCAL],
            },
            'eval_split': {
                'type': 'str',
                'description': 'Source split to load; defaults to test.',
                'value': '',
            },
            'dataset_revision': {
                'type': 'str',
                'description': 'Optional dataset revision; leave empty to use the hub default.',
                'value': '',
            },
            'video_dir': {
                'type': 'str',
                'description': 'Optional local directory containing MSVD video files.',
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
class MSVDAdapter(CaptionDatasetAdapter):
    """Adapter for MSVD video captioning."""

    media_fields = ['video', 'video_path', 'url']
    reference_fields = ['references', 'caption', 'captions']
    group_key_field = 'video_id'
    source_dataset_ids = {
        HubType.HUGGINGFACE: 'VLM2Vec/MSVD',
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
