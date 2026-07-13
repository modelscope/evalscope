# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, Sample, build_dataset_from_records
from evalscope.api.messages import ChatMessageUser, Content, ContentText, ContentVideo
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, prompt
from evalscope.utils.url_utils import guess_video_format
from .utils import build_question, find_archive_member, optional_float

logger = get_logger()


@dataclass(frozen=True)
class MVBenchSubsetSpec:
    archive: str


SUBSET_SPECS = {
    'action_antonym': MVBenchSubsetSpec(archive='ssv2_video.zip'),
    'action_count': MVBenchSubsetSpec(archive='perception.zip'),
    'action_localization': MVBenchSubsetSpec(archive='sta.zip'),
    'action_prediction': MVBenchSubsetSpec(archive='star.zip'),
    'action_sequence': MVBenchSubsetSpec(archive='star.zip'),
    'character_order': MVBenchSubsetSpec(archive='perception.zip'),
    'counterfactual_inference': MVBenchSubsetSpec(archive='clevrer.zip'),
    'egocentric_navigation': MVBenchSubsetSpec(archive='vlnqa.zip'),
    'episodic_reasoning': MVBenchSubsetSpec(archive='tvqa.zip'),
    'fine_grained_action': MVBenchSubsetSpec(archive='Moments_in_Time_Raw.zip'),
    'fine_grained_pose': MVBenchSubsetSpec(archive='nturgbd.zip'),
    'moving_attribute': MVBenchSubsetSpec(archive='clevrer.zip'),
    'moving_count': MVBenchSubsetSpec(archive='clevrer.zip'),
    'moving_direction': MVBenchSubsetSpec(archive='clevrer.zip'),
    'object_existence': MVBenchSubsetSpec(archive='clevrer.zip'),
    'object_interaction': MVBenchSubsetSpec(archive='star.zip'),
    'object_shuffle': MVBenchSubsetSpec(archive='perception.zip'),
    'scene_transition': MVBenchSubsetSpec(archive='scene_qa.zip'),
    'state_change': MVBenchSubsetSpec(archive='perception.zip'),
    'unexpected_action': MVBenchSubsetSpec(archive='FunQA_test.zip'),
}

SUBSET_LIST = list(SUBSET_SPECS)
DEFAULT_SUBSET_LIST = ['action_antonym']


@register_benchmark(
    BenchmarkMeta(
        name='mvbench',
        pretty_name='MVBench',
        description="""
## Overview

MVBench is a public multimodal video understanding benchmark covering temporal perception,
attribute/state reasoning, symbolic ordering, and high-level cognition. This native adapter uses
the ModelScope `PKU-Alignment/MVBench` mirror by default, which provides JSON annotations plus
optimized video archives.

## Task Description

- **Task Type**: Video multiple-choice question answering
- **Input**: Video + question + answer choices
- **Output**: Single correct answer letter
- **Subsets**: 20 MVBench tasks; the default smoke-test subset is `action_antonym`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- The default `action_antonym` subset downloads a small public MP4 archive for quick validation
- Full benchmark evaluation can be requested by setting `subset_list` to additional MVBench subsets
- Time-bounded records keep start/end metadata and add a short segment instruction to the prompt
""",
        tags=[Tags.MULTI_MODAL, Tags.VIDEO, Tags.MULTIPLE_CHOICE],
        dataset_id='PKU-Alignment/MVBench',
        paper_url='https://arxiv.org/abs/2311.17005',
        subset_list=DEFAULT_SUBSET_LIST,
        metric_list=['acc'],
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MVBenchAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    """Native adapter for the public MVBench video multiple-choice benchmark."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._video_cache_dir: Optional[str] = None

    @property
    def video_cache_dir(self) -> str:
        if self._video_cache_dir is None:
            self._video_cache_dir = os.path.join(self.dataset_dir, 'mvbench', 'videos')
        return self._video_cache_dir

    @property
    def source_dataset(self) -> DatasetHub:
        return DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.dataset_hub,
            force_redownload=self.force_redownload,
        )

    def load(self) -> Tuple[DatasetDict, None]:
        dataset_dict = {}
        for subset in self.subset_list:
            if subset not in SUBSET_LIST:
                raise ValueError(f'Unsupported MVBench subset: {subset}. Supported subsets: {SUBSET_LIST}')
            with self._temporary_attribute('current_subset_name', subset):
                records = self._load_subset_records(subset)
                dataset = build_dataset_from_records(
                    records=records,
                    sample_fields=self.record_to_sample,
                    name='mvbench',
                    location=self.dataset_id,
                    limit=self.limit,
                    repeats=self.repeats,
                    shuffle=self.shuffle,
                    seed=self.seed,
                )
                dataset_dict[subset] = dataset

        return DatasetDict(dataset_dict), None

    def _load_subset_records(self, subset: str) -> List[Dict[str, Any]]:
        logger.info(f'Loading MVBench subset {subset} from {self.dataset_hub}: {self.dataset_id}.')
        dataset = self.source_dataset.load(split=self.eval_split, subset=subset)
        records = list(dataset)
        if not records:
            raise ValueError(f'No records found for MVBench subset: {subset}')
        return records

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [str(choice) for choice in record['candidates']]
        answer = str(record['answer'])
        target = answer_character(choices.index(answer))
        start = optional_float(record.get('start'), 'start')
        end = optional_float(record.get('end'), 'end')
        fps = optional_float(record.get('fps'), 'fps')
        if start is not None and end is not None and start > end:
            raise ValueError(f'Invalid MVBench time boundary: start={start} is greater than end={end}.')

        video_path = self._ensure_video_file(self.current_subset_name, str(record['video']))
        input_text = prompt(
            question=build_question(record, start=start, end=end),
            choices=choices,
            template=self.prompt_template,
        )
        content_list: List[Content] = [
            ContentText(text=input_text),
            ContentVideo(video=video_path, format=guess_video_format(video_path), start=start, end=end, fps=fps),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=choices,
            target=target,
            metadata={
                'video': record.get('video'),
                'answer': answer,
                'subset': self.current_subset_name,
                'start': start,
                'end': end,
                'fps': fps,
                'dataset_id': self.dataset_id,
                'dataset_hub': self.dataset_hub,
            },
        )

    def _ensure_video_file(self, subset: str, video_name: str) -> str:
        archive_name = SUBSET_SPECS[subset].archive
        output_path = self._cache_output_path(subset, video_name)
        if os.path.exists(output_path) and not self.force_redownload:
            return output_path

        archive_path = self.source_dataset.download_file(f'video/{archive_name}')
        member_name = find_archive_member(archive_path, subset, video_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with zipfile.ZipFile(archive_path) as zip_file:
            with zip_file.open(member_name) as source, open(output_path, 'wb') as target:
                target.write(source.read())
        return output_path

    def _cache_output_path(self, subset: str, video_name: str) -> str:
        subset_dir = os.path.abspath(os.path.join(self.video_cache_dir, subset))
        output_path = os.path.abspath(os.path.join(subset_dir, video_name))
        if os.path.commonpath([subset_dir, output_path]) != subset_dir:
            raise ValueError(f'Invalid MVBench video path: {video_name}')
        return output_path
