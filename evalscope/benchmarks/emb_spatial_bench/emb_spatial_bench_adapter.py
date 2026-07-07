# flake8: noqa: E501
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample, download_dataset_file
from evalscope.api.dataset.dataset import DatasetDict, MemoryDataset
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt

logger = get_logger()

DESCRIPTION = """
## Overview

EmbSpatial-Bench is a benchmark for evaluating embodied spatial understanding of large vision-language models (LVLMs). \
The benchmark is automatically derived from embodied scenes and covers 6 spatial relationships from an egocentric \
perspective: **close**, **far**, **above**, **under**, **left**, and **right**.

## Task Description

- **Task Type**: Multiple-Choice Visual Question Answering (VQA)
- **Input**: An egocentric RGB image + a spatial reasoning question with 4 candidate answers
- **Output**: A single letter (A / B / C / D) identifying the correct object or spatial relationship
- **Domains**: Embodied AI, spatial reasoning (MP3D and AI2Thor environments)

## Key Features

- 3,640 human-verified evaluation questions derived from two embodied environments (MP3D and AI2Thor)
- 6 spatial relation categories: close, far, above, under, left, right
- Each question requires selecting the most spatially accurate answer from 4 options
- Designed to expose the gap between current LVLMs and qualified embodied intelligence

## Evaluation Notes

- Default evaluation uses the **embspatial_bench.json** file (3,640 samples)
- Primary metric: **Accuracy** (acc)
- Answer indices are 0-based in the dataset (0 → A, 1 → B, 2 → C, 3 → D)
- Images are stored as JPEG base64 strings in the JSON file
- Subsets are organized by the `relation` field (6 spatial categories)
- [Paper](https://aclanthology.org/2024.acl-short.33/) | [GitHub](https://github.com/mengfeidu/EmbSpatial-Bench)
"""


@register_benchmark(
    BenchmarkMeta(
        name='emb_spatial_bench',
        pretty_name='EmbSpatial-Bench',
        dataset_id='evalscope/EmbSpatial-Bench',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION,
        paper_url='https://aclanthology.org/2024.acl-short.33/',
        subset_list=['close', 'far', 'above', 'under', 'left', 'right'],
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class EmbSpatialBenchAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    """Data adapter for evalscope/EmbSpatial-Bench.

    Downloads embspatial_bench.json (skipping the large SFT file), converts each
    record into a multimodal Sample with a base64-encoded JPEG image, and evaluates
    spatial MCQ accuracy.  Subsets correspond to the 6 spatial relations.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load(self) -> Tuple[DatasetDict, None]:
        """Download embspatial_bench.json and build a DatasetDict split by relation."""
        data_source = HubType.LOCAL if os.path.exists(self.dataset_id) else self.dataset_hub
        bench_json = download_dataset_file(
            data_id_or_path=self.dataset_id,
            file_path='embspatial_bench.json',
            data_source=data_source,
        )
        logger.info(f'Loading EmbSpatial-Bench records from {bench_json}')
        with open(bench_json, 'r', encoding='utf-8') as fh:
            records: List[Dict[str, Any]] = json.load(fh)

        samples: List[Sample] = []
        for rec in records:
            s = self.record_to_sample(rec)
            if s is not None:
                samples.append(s)

        mem_dataset = MemoryDataset(samples, name='emb_spatial_bench')

        dataset_dict = DatasetDict.from_dataset(
            dataset=mem_dataset,
            subset_list=self.subset_list,
            limit=self.limit,
            repeats=self.repeats,
        )
        return dataset_dict, None

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Optional[Sample]:
        """Convert a raw EmbSpatial-Bench JSON record to a multimodal Sample.

        The ``image`` field is a raw JPEG base64 string (no data-URI header).
        The ``answer`` field is a 0-based index into ``answer_options``.
        """
        image_b64_raw: str = record.get('image', '')
        if not image_b64_raw:
            logger.warning(f'Record {record.get("question_id")} has no image; skipping.')
            return None

        # Prepend data-URI header so the model API accepts it
        image_b64 = f'data:image/jpeg;base64,{image_b64_raw}'

        question: str = record.get('question', '')
        answer_options: List[str] = record.get('answer_options', [])
        answer_idx: int = int(record.get('answer', 0))

        # Use framework MCQ prompt formatting
        input_text = prompt(question=question, choices=answer_options, template=self.prompt_template)

        content_list: List[Content] = [
            ContentImage(image=image_b64),
            ContentText(text=input_text),
        ]

        # 0-indexed answer -> letter (0->A, 1->B, 2->C, 3->D)
        target = chr(65 + answer_idx) if 0 <= answer_idx <= 25 else ''

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answer_options,
            target=target,
            subset_key=record.get('relation', ''),
            metadata={
                'question_id': record.get('question_id', ''),
                'relation': record.get('relation', ''),
                'data_source': record.get('data_source', ''),
            },
        )
