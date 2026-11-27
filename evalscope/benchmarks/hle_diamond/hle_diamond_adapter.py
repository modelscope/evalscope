from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.hle.hle_adapter import HLEAdapter
from evalscope.constants import Tags

SUBSET_LIST = ['reasoning', 'knowledge']


@register_benchmark(
    BenchmarkMeta(
        name='hle_diamond',
        pretty_name="Humanity's-Last-Exam-Diamond",
        evaluation_version='v1.0',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.REASONING],
        description="""
## Overview

Humanity's Last Exam Diamond (HLE-Diamond) is the official refined 1,000-question subset of Humanity's Last Exam. It retains the original expert-level question format while providing separate reasoning and knowledge partitions.

## Task Description

- **Task Type**: Expert-Level Question Answering
- **Input**: Question with an optional image
- **Output**: Answer with explanation and confidence score
- **Domain**: Broad academic and professional disciplines, with reasoning and knowledge partitions

## Key Features

- 1,000 questions: 500 reasoning and 500 knowledge questions
- Uses the original HLE answer types and multimodal questions
- Reports overall accuracy and separate reasoning/knowledge partition results
- Preserves the original question metadata, including subject category and rationale

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with the same JSON C/I LLM judge contract as HLE
- Response format includes: Explanation, Answer, and Confidence (0-100%)
- Set `extra_params["include_multi_modal"]` to `False` for text-only models
- Hugging Face access requires accepting the dataset conditions and providing an access token; an authenticated or local dataset path is also supported
""",  # noqa: E501
        dataset_id='cais/hle-diamond',
        hf_dataset_id='cais/hle-diamond',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params={
            'include_multi_modal': {
                'type': 'bool',
                'description': 'Include multi-modal (image) questions during evaluation.',
                'value': True,
            }
        },
    )
)
class HLEDiamondAdapter(HLEAdapter):
    _suppress_doc_sample_example = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        sample = super().record_to_sample(record)
        partition = record['partition']
        if partition not in SUBSET_LIST:
            raise ValueError(f'Unsupported HLE-Diamond partition {partition!r}; expected one of {SUBSET_LIST}.')
        sample.subset_key = partition
        sample.metadata['partition'] = partition
        return sample
