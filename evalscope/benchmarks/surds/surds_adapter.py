import json
import os
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, Sample, build_dataset_dict_from_record_map
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import (
    IMAGE_SIZE,
    PAIRED_SUBSETS,
    SUBSET_LIST,
    build_official_vqa_records,
    compute_centerness,
    extract_tagged_answer,
    normalize_answer,
    option_is_present,
    parse_point,
)

DESCRIPTION = """
## Overview

SURDS benchmarks fine-grained spatial understanding and reasoning by vision-language models in realistic driving
scenes. It is derived from the six-camera nuScenes dataset and evaluates object-centric and relational spatial skills
without supplying depth maps or visual markers.

## Task Description

- **Task Type**: Multi-task visual spatial question answering
- **Input**: A 1600 x 900 driving-scene image and an English spatial reasoning question
- **Output**: A structured response ending in an answer inside `<answer>...</answer>`
- **Domain**: Autonomous driving and outdoor 3D spatial reasoning

## Key Features

- 9,250 model queries generated deterministically from 5,919 validation images, following the official seed-42 code
- Six equally weighted task subsets: yaw orientation, pixel localization, depth range, pairwise distance, left/right
  ordering, and front/behind relation
- Yaw, distance, left/right, and front/behind are consistency tests: both complementary prompts for an evaluation unit
  must be correct to receive credit
- Images come from six nuScenes cameras and contain unmarked objects described by appearance rather than overlays

## Evaluation Notes

- The official prompts and `<think>...<answer>...</answer>` response contract are reproduced verbatim
- Pixel localization uses the official centerness metric: predictions outside the target box receive 0, while points
  nearer the box center receive scores approaching 1; normalized coordinates and predicted boxes are also accepted
- The other five tasks use official normalized exact match, removing case, punctuation, articles, and extra whitespace
- Every subset contains 925 evaluation units; the overall normalized score is therefore the equal average of all six
  task scores. A full run makes 9,250 model requests but reports `Num=5,550`, because each complementary prompt pair
  is one official evaluation unit
- Invalid or missing `<answer>` blocks score 0, matching the official benchmark denominator semantics
- The dataset is evaluation-only and downloaded from ModelScope; only images needed by the selected subsets are fetched
- Resources: [Paper](https://arxiv.org/abs/2411.13112) |
  [GitHub](https://github.com/XiandaGuo/Drive-MLLM)
"""


@register_benchmark(
    BenchmarkMeta(
        name='surds',
        pretty_name='SURDS',
        dataset_id='evalscope/SURDS_eval',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.QA, Tags.GROUNDING],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2411.13112',
        subset_list=SUBSET_LIST,
        metric_list=['normalized_score'],
        primary_metric='normalized_score',
        eval_split='validation',
        evaluation_version='v1.0',
    )
)
class SURDSAdapter(VisionLanguageAdapter):
    """Official SURDS VQA generation and scoring over the evaluation-only image metadata."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._split_root: Optional[str] = None

    def load(self) -> Tuple[DatasetDict, None]:
        """Generate the six official VQA subsets from metadata and resolve their images."""
        unknown = [subset for subset in self.subset_list if subset not in SUBSET_LIST]
        if unknown:
            raise ValueError(f'Unknown SURDS subsets {unknown}. Valid subsets are: {SUBSET_LIST}')

        hub = DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        )
        metadata_relpath = f'{self.eval_split}/metadata.jsonl'
        metadata_path = hub.download_file(metadata_relpath)
        self._split_root = os.path.dirname(metadata_path)
        with open(metadata_path, 'r', encoding='utf-8') as file:
            source_records = [json.loads(line) for line in file if line.strip()]

        all_records = build_official_vqa_records(source_records)
        record_map = {subset: all_records[subset] for subset in self.subset_list}
        datasets = build_dataset_dict_from_record_map(
            record_map=record_map,
            sample_fields=self.record_to_sample,
            location=self.dataset_id,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        )

        dataset_root = os.path.dirname(self._split_root)
        image_relpaths = sorted(
            {
                os.path.relpath(sample.metadata['image_path'], dataset_root)
                for dataset in datasets.values()
                for sample in dataset
            }
        )
        missing_images = [path for path in image_relpaths if not os.path.isfile(os.path.join(dataset_root, path))]
        if self.force_redownload or missing_images:
            download_images = image_relpaths if self.force_redownload else missing_images
            hub.download_snapshot(allow_file_pattern=[metadata_relpath, *download_images])
            missing_images = [path for path in image_relpaths if not os.path.isfile(os.path.join(dataset_root, path))]
        if missing_images:
            raise FileNotFoundError(f'SURDS snapshot is missing {len(missing_images)} required images.')

        return datasets, None

    def record_to_sample(self, record: Dict[str, Any]) -> List[Sample]:
        """Expand one official evaluation unit into one or two model prompts."""
        if self._split_root is None:
            raise RuntimeError('SURDS dataset must be loaded before converting records.')

        image_path = os.path.join(self._split_root, record['image_path'])
        paired = record['task'] in PAIRED_SUBSETS
        samples: List[Sample] = []
        for variant_index, (prompt, answer) in enumerate(zip(record['prompts'], record['answers'])):
            content: List[Content] = [ContentText(text=prompt), ContentImage(image=image_path)]
            samples.append(
                Sample(
                    input=[ChatMessageUser(content=content)],
                    target=answer,
                    metadata={
                        'task': record['task'],
                        'pair_id': record['pair_id'],
                        'variant_index': variant_index,
                        'paired': paired,
                        'bbox': record['bbox'],
                        'options': record['options'],
                        'image_size': list(IMAGE_SIZE),
                        'image_path': image_path,
                    },
                )
            )
        return samples

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract the official tagged final answer."""
        return extract_tagged_answer(prediction)

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score one response with exact match or pixel centerness."""
        metadata = task_state.metadata or {}
        if metadata.get('task') == 'xy2d':
            point = parse_point(filtered_prediction, tuple(metadata['image_size'])) if filtered_prediction else None
            value = compute_centerness(point, metadata['bbox']) if point is not None else 0.0
        else:
            options = metadata.get('options', [])
            valid = not options or any(option_is_present(option, filtered_prediction) for option in options)
            value = float(
                bool(filtered_prediction)
                and valid
                and normalize_answer(filtered_prediction) == normalize_answer(reference)
            )

        return Score(
            value={'normalized_score': value},
            main_score_name='normalized_score',
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Apply official pairwise consistency aggregation to complementary-prompt tasks."""
        if not sample_scores or not (sample_scores[0].sample_metadata or {}).get('paired'):
            return super().aggregate_scores(sample_scores)

        grouped: Dict[Tuple[str, int], List[SampleScore]] = defaultdict(list)
        for sample_score in sample_scores:
            metadata = sample_score.sample_metadata or {}
            generation_index = sample_score.generation_index if sample_score.generation_index is not None else 0
            grouped[(str(metadata['pair_id']), generation_index)].append(sample_score)

        values: List[float] = []
        ids: List[str] = []
        for (pair_id, generation_index), pair_scores in grouped.items():
            pair_value = 0.0
            if len(pair_scores) == 2:
                pair_value = float(all(score.score.value.get('normalized_score', 0.0) == 1.0 for score in pair_scores))
            values.append(pair_value)
            ids.append(f'{pair_id}:{generation_index}')

        return [
            AggScore(
                score=mean(values),
                metric_name='normalized_score',
                aggregation='mean',
                num=len(values),
                ids=ids,
                metadata={'unit': 'complementary_prompt_pair'},
            )
        ]
