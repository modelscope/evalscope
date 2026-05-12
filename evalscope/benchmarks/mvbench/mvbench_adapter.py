# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os
import random
import zipfile
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentText, ContentVideo
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, prompt
from evalscope.utils.url_utils import guess_video_format

logger = get_logger()

SUBSET_LIST = [
    'action_antonym',
    'action_count',
    'action_localization',
    'action_prediction',
    'action_sequence',
    'character_order',
    'counterfactual_inference',
    'egocentric_navigation',
    'episodic_reasoning',
    'fine_grained_action',
    'fine_grained_pose',
    'moving_attribute',
    'moving_count',
    'moving_direction',
    'object_existence',
    'object_interaction',
    'object_shuffle',
    'scene_transition',
    'state_change',
    'unexpected_action',
]

DEFAULT_SUBSET_LIST = ['action_antonym']

SUBSET_ARCHIVES = {
    'action_antonym': 'ssv2_video.zip',
    'action_count': 'perception.zip',
    'action_localization': 'sta.zip',
    'action_prediction': 'star.zip',
    'action_sequence': 'star.zip',
    'character_order': 'perception.zip',
    'counterfactual_inference': 'clevrer.zip',
    'egocentric_navigation': 'vlnqa.zip',
    'episodic_reasoning': 'tvqa.zip',
    'fine_grained_action': 'Moments_in_Time_Raw.zip',
    'fine_grained_pose': 'nturgbd.zip',
    'moving_attribute': 'clevrer.zip',
    'moving_count': 'clevrer.zip',
    'moving_direction': 'clevrer.zip',
    'object_existence': 'clevrer.zip',
    'object_interaction': 'star.zip',
    'object_shuffle': 'perception.zip',
    'scene_transition': 'scene_qa.zip',
    'state_change': 'perception.zip',
    'unexpected_action': 'FunQA_test.zip',
}


@register_benchmark(
    BenchmarkMeta(
        name='mvbench',
        pretty_name='MVBench',
        description="""
## Overview

MVBench is a public multimodal video understanding benchmark covering temporal perception,
attribute/state reasoning, symbolic ordering, and high-level cognition. This native adapter uses
the Hugging Face `PKU-Alignment/MVBench` mirror, which provides JSON annotations plus optimized
video archives.

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
""",
        tags=[Tags.MULTI_MODAL, Tags.MULTIPLE_CHOICE],
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
                dataset = MemoryDataset(samples=samples, name='mvbench', location=self.dataset_id)
                dataset.reindex(group_size=self.repeats)
                dataset_dict[subset] = dataset

        self.test_dataset = DatasetDict(dataset_dict)
        self.fewshot_dataset = None
        self._post_process_samples()
        return self.test_dataset

    def _load_subset_records(self, subset: str) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        logger.info(f'Loading MVBench subset {subset} from {self.dataset_id}.')
        dataset = load_dataset(self.dataset_id, subset, split=self.eval_split)
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
        video_path = self._ensure_video_file(self.current_subset_name, str(record['video']))
        input_text = prompt(question=record['question'], choices=choices, template=self.prompt_template)
        content_list: List[Content] = [
            ContentText(text=input_text),
            ContentVideo(video=video_path, format=guess_video_format(video_path)),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=choices,
            target=target,
            metadata={
                'video': record.get('video'),
                'answer': answer,
                'subset': self.current_subset_name,
                'start': record.get('start'),
                'end': record.get('end'),
            },
        )

    def _ensure_video_file(self, subset: str, video_name: str) -> str:
        archive_name = SUBSET_ARCHIVES[subset]
        output_path = os.path.join(self.video_cache_dir, subset, video_name)
        if os.path.exists(output_path) and not self.force_redownload:
            return output_path

        from huggingface_hub import hf_hub_download

        archive_path = hf_hub_download(self.dataset_id, f'video/{archive_name}', repo_type='dataset')
        member_name = self._find_archive_member(archive_path, subset, video_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with zipfile.ZipFile(archive_path) as zip_file:
            with zip_file.open(member_name) as source, open(output_path, 'wb') as target:
                target.write(source.read())
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
