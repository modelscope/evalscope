# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os
import random
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, MemoryDataset, Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentText, ContentVideo
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, prompt
from evalscope.utils.url_utils import guess_video_format

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
        tags=[Tags.MULTI_MODAL, Tags.MULTIPLE_CHOICE],
        dataset_id='PKU-Alignment/MVBench',
        paper_url='https://arxiv.org/abs/2311.17005',
        subset_list=DEFAULT_SUBSET_LIST,
        metric_list=['acc'],
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
        extra_params={
            'dataset_id': {
                'type': 'str',
                'description': 'Dataset repository ID or local dataset root for MVBench annotations and videos.',
                'value': 'PKU-Alignment/MVBench',
            },
            'dataset_hub': {
                'type': 'str',
                'description': 'Dataset hub used to load annotations and video archives.',
                'value': HubType.MODELSCOPE,
                'choices': [HubType.HUGGINGFACE, HubType.MODELSCOPE, HubType.LOCAL],
            },
            'dataset_revision': {
                'type': 'str',
                'description': 'Optional dataset revision; leave empty to use the hub default.',
                'value': '',
            },
        },
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
    def source_dataset_id(self) -> str:
        return self.extra_params.get('dataset_id') or self.dataset_id

    @property
    def source_dataset_hub(self) -> str:
        return self.extra_params.get('dataset_hub') or HubType.MODELSCOPE

    @property
    def source_dataset_revision(self) -> Optional[str]:
        return self.extra_params.get('dataset_revision') or None

    @property
    def source_dataset(self) -> DatasetHub:
        return DatasetHub(
            data_id_or_path=self.source_dataset_id,
            data_source=self.source_dataset_hub,
            revision=self.source_dataset_revision,
            force_redownload=self.force_redownload,
        )

    def load_dataset(self) -> DatasetDict:
        dataset_dict: Dict[str, MemoryDataset] = {}
        for subset in self.subset_list:
            if subset not in SUBSET_LIST:
                raise ValueError(f'Unsupported MVBench subset: {subset}. Supported subsets: {SUBSET_LIST}')
            with self._temporary_attribute('current_subset_name', subset):
                records = self._load_subset_records(subset)
                if self.shuffle:
                    random.Random(self.seed).shuffle(records)
                records = self._apply_limit(records)
                samples = [self.record_to_sample(record) for record in records]
                if self.repeats > 1:
                    samples = [copy.deepcopy(sample) for sample in samples for _ in range(self.repeats)]
                dataset = MemoryDataset(samples=samples, name='mvbench', location=self.source_dataset_id)
                dataset.reindex(group_size=self.repeats)
                dataset_dict[subset] = dataset

        self.test_dataset = DatasetDict(dataset_dict)
        self.fewshot_dataset = None
        self._post_process_samples()
        return self.test_dataset

    def _load_subset_records(self, subset: str) -> List[Dict[str, Any]]:
        logger.info(f'Loading MVBench subset {subset} from {self.source_dataset_hub}: {self.source_dataset_id}.')
        dataset = self.source_dataset.load(split=self.eval_split, subset=subset)
        records = list(dataset)
        if not records:
            raise ValueError(f'No records found for MVBench subset: {subset}')
        return records

    def _apply_limit(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.limit is None:
            return records
        if isinstance(self.limit, float):
            limit = int(len(records) * self.limit)
        else:
            limit = self.limit
        return records[:limit]

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [str(choice) for choice in record['candidates']]
        answer = str(record['answer'])
        target = answer_character(choices.index(answer))
        start = self._optional_float(record.get('start'), 'start')
        end = self._optional_float(record.get('end'), 'end')
        fps = self._optional_float(record.get('fps'), 'fps')
        if start is not None and end is not None and start > end:
            raise ValueError(f'Invalid MVBench time boundary: start={start} is greater than end={end}.')

        video_path = self._ensure_video_file(self.current_subset_name, str(record['video']))
        input_text = prompt(
            question=self._question_with_video_context(record, start=start, end=end),
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
                'dataset_id': self.source_dataset_id,
                'dataset_hub': self.source_dataset_hub,
            },
        )

    def _ensure_video_file(self, subset: str, video_name: str) -> str:
        archive_name = SUBSET_SPECS[subset].archive
        output_path = self._cache_output_path(subset, video_name)
        if os.path.exists(output_path) and not self.force_redownload:
            return output_path

        archive_path = self.source_dataset.download_file(f'video/{archive_name}')
        member_name = self._find_archive_member(archive_path, subset, video_name)
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

    @staticmethod
    def _find_archive_member(archive_path: str, subset: str, video_name: str) -> str:
        normalized_subset = subset.replace('_', '').lower()
        with zipfile.ZipFile(archive_path) as zip_file:
            matches = [
                name for name in zip_file.namelist()
                if not name.endswith('/') and (name.endswith(f'/{video_name}') or name.endswith(video_name))
            ]
        if not matches:
            raise FileNotFoundError(f'Video {video_name} was not found in archive {archive_path}.')

        preferred = [name for name in matches if normalized_subset in name.replace('_', '').lower()]
        return sorted(preferred or matches)[0]

    def _question_with_video_context(
        self,
        record: Dict[str, Any],
        start: Optional[float],
        end: Optional[float],
    ) -> str:
        context = self._format_video_context(record, start=start, end=end)
        question = str(record['question'])
        if not context:
            return question
        return f'{context}\n\n{question}'

    @classmethod
    def _format_video_context(cls, record: Dict[str, Any], start: Optional[float], end: Optional[float]) -> str:
        context_parts = []
        time_range = cls._format_time_range(start, end)
        if time_range:
            context_parts.append(f'Answer based on the video segment {time_range}.')

        subtitle = record.get('subtitle') or record.get('subtitles')
        if subtitle:
            context_parts.append(f'Subtitles:\n{subtitle}')

        return '\n'.join(context_parts)

    @classmethod
    def _format_time_range(cls, start: Optional[float], end: Optional[float]) -> str:
        if start is None and end is None:
            return ''
        if start is not None and end is not None:
            return f'from {cls._format_seconds(start)}s to {cls._format_seconds(end)}s'
        if start is not None:
            return f'after {cls._format_seconds(start)}s'
        return f'before {cls._format_seconds(end)}s'

    @staticmethod
    def _format_seconds(value: float) -> str:
        return f'{value:g}'

    @staticmethod
    def _optional_float(value: Any, field_name: str) -> Optional[float]:
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid MVBench {field_name}: {value!r}') from exc
