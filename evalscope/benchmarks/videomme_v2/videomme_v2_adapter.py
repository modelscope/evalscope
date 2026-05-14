# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import json
import os
import random
import re
import zipfile
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, MemoryDataset, Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentText, ContentVideo
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt
from evalscope.utils.url_utils import guess_video_format

logger = get_logger()

DATASET_ID = 'MME-Benchmarks/Video-MME-v2'
SUBSET_LIST = ['all', 'level_1', 'level_2', 'level_3', 'logic', 'relevance']
DEFAULT_SUBSET_LIST = ['all']
VIDEO_SOURCE_URL = 'url'
VIDEO_SOURCE_ARCHIVE = 'archive'
VIDEO_SOURCE_LIST = [VIDEO_SOURCE_URL, VIDEO_SOURCE_ARCHIVE]
OPTION_PATTERN = re.compile(r'^\s*([A-Z])[\.\)]\s*(.*)$')


@register_benchmark(
    BenchmarkMeta(
        name='videomme_v2',
        pretty_name='Video-MME-v2',
        description="""
## Overview

Video-MME-v2 is a public comprehensive video understanding benchmark. It contains 800 videos,
3,200 multiple-choice QA instances, and word-level subtitles with timestamps. The native adapter
uses the shared `DatasetHub` abstraction for both annotation loading and optional media archive
downloads, so it exercises the same reusable video benchmark path as MVBench.

## Task Description

- **Task Type**: Video multiple-choice question answering
- **Input**: Video URL or archived MP4 + question + answer choices
- **Output**: Single correct answer letter
- **Subsets**: `all`, `level_1`, `level_2`, `level_3`, `logic`, `relevance`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- The default video source is the public `url` field for lightweight smoke tests
- Set `extra_params.video_source` to `archive` to download and use the official MP4 archives
- Set `extra_params.use_subtitles` to `true` to include word-level subtitles in the prompt
""",
        tags=[Tags.MULTI_MODAL, Tags.MULTIPLE_CHOICE],
        dataset_id=DATASET_ID,
        paper_url='https://arxiv.org/abs/2604.05015',
        subset_list=DEFAULT_SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
        extra_params={
            'dataset_id': {
                'type': 'str',
                'description': 'Dataset repository ID or local dataset root for Video-MME-v2.',
                'value': DATASET_ID,
            },
            'dataset_hub': {
                'type': 'str',
                'description': 'Dataset hub used to load annotations, subtitles, and optional video archives.',
                'value': HubType.MODELSCOPE,
                'choices': [HubType.HUGGINGFACE, HubType.MODELSCOPE, HubType.LOCAL],
            },
            'dataset_revision': {
                'type': 'str',
                'description': 'Optional dataset revision; leave empty to use the hub default.',
                'value': '',
            },
            'video_source': {
                'type': 'str',
                'description': 'Use public URL fields for lightweight tests or official archived MP4 files.',
                'value': VIDEO_SOURCE_URL,
                'choices': VIDEO_SOURCE_LIST,
            },
            'use_subtitles': {
                'type': 'bool',
                'description': 'Include Video-MME-v2 subtitle text in the prompt.',
                'value': False,
            },
            'subtitle_word_limit': {
                'type': 'int',
                'description': 'Maximum number of subtitle words included per sample when subtitles are enabled.',
                'value': 512,
            },
        },
    )
)
class VideoMMEv2Adapter(VisionLanguageAdapter, MultiChoiceAdapter):
    """Native adapter for the public Video-MME-v2 video multiple-choice benchmark."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._video_cache_dir: Optional[str] = None
        self._record_cache: Optional[List[Dict[str, Any]]] = None
        self._subtitle_cache: Dict[str, str] = {}

    @property
    def video_cache_dir(self) -> str:
        if self._video_cache_dir is None:
            self._video_cache_dir = os.path.join(self.dataset_dir, 'videomme_v2', 'videos')
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

    @property
    def video_source(self) -> str:
        return self.extra_params.get('video_source') or VIDEO_SOURCE_URL

    @property
    def use_subtitles(self) -> bool:
        return bool(self.extra_params.get('use_subtitles'))

    @property
    def subtitle_word_limit(self) -> int:
        return int(self.extra_params.get('subtitle_word_limit') or 0)

    def load_dataset(self) -> DatasetDict:
        dataset_dict: Dict[str, MemoryDataset] = {}
        for subset in self.subset_list:
            if subset not in SUBSET_LIST:
                raise ValueError(f'Unsupported Video-MME-v2 subset: {subset}. Supported subsets: {SUBSET_LIST}')
            with self._temporary_attribute('current_subset_name', subset):
                records = self._records_for_subset(subset)
                if self.shuffle:
                    random.Random(self.seed).shuffle(records)
                records = self._apply_limit(records)
                samples = [self.record_to_sample(record) for record in records]
                if self.repeats > 1:
                    samples = [copy.deepcopy(sample) for sample in samples for _ in range(self.repeats)]
                dataset = MemoryDataset(samples=samples, name='videomme_v2', location=self.source_dataset_id)
                dataset.reindex(group_size=self.repeats)
                dataset_dict[subset] = dataset

        self.test_dataset = DatasetDict(dataset_dict)
        self.fewshot_dataset = None
        self._post_process_samples()
        return self.test_dataset

    def _load_records(self) -> List[Dict[str, Any]]:
        if self._record_cache is None:
            logger.info(f'Loading Video-MME-v2 records from {self.source_dataset_hub}: {self.source_dataset_id}.')
            dataset = self.source_dataset.load(split=self.eval_split)
            self._record_cache = list(dataset)
            if not self._record_cache:
                raise ValueError('No records found for Video-MME-v2.')
        return list(self._record_cache)

    def _records_for_subset(self, subset: str) -> List[Dict[str, Any]]:
        records = self._load_records()
        if subset == 'all':
            return records
        if subset.startswith('level_'):
            level = subset.rsplit('_', 1)[-1]
            return [record for record in records if str(record.get('level')) == level]
        return [record for record in records if record.get('group_type') == subset]

    def _apply_limit(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.limit is None:
            return records
        if isinstance(self.limit, float):
            limit = int(len(records) * self.limit)
        else:
            limit = self.limit
        return records[:limit]

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = self._parse_options(record['options'])
        target = self._normalize_answer(record['answer'], choices)
        video_id = str(record['video_id'])
        video = self._resolve_video(record)
        subtitle = self._subtitle_for_record(video_id) if self.use_subtitles else None
        input_text = prompt(
            question=self._question_with_context(record, subtitle=subtitle),
            choices=choices,
            template=self.prompt_template,
        )
        content_list: List[Content] = [
            ContentText(text=input_text),
            ContentVideo(video=video, format=guess_video_format(video)),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=choices,
            target=target,
            metadata={
                'video_id': video_id,
                'url': record.get('url'),
                'question_id': record.get('question_id'),
                'group_type': record.get('group_type'),
                'group_structure': record.get('group_structure'),
                'level': record.get('level'),
                'second_head': record.get('second_head'),
                'third_head': record.get('third_head'),
                'answer': target,
                'subset': self.current_subset_name,
                'video_source': self.video_source,
                'dataset_id': self.source_dataset_id,
                'dataset_hub': self.source_dataset_hub,
            },
        )

    @classmethod
    def _parse_options(cls, options: Any) -> List[str]:
        if isinstance(options, list):
            choices = [str(option).strip() for option in options]
        else:
            choices = []
            for line in str(options).splitlines():
                line = line.strip()
                if not line:
                    continue
                match = OPTION_PATTERN.match(line)
                if match:
                    choices.append(match.group(2).strip())
                elif choices:
                    choices[-1] = f'{choices[-1]} {line}'
                else:
                    choices.append(line)
        if not choices:
            raise ValueError(f'Invalid Video-MME-v2 options: {options!r}')
        return choices

    @staticmethod
    def _normalize_answer(answer: Any, choices: List[str]) -> str:
        target = str(answer).strip().upper()
        valid_answers = {chr(ord('A') + idx) for idx in range(len(choices))}
        if target not in valid_answers:
            raise ValueError(f'Invalid Video-MME-v2 answer {answer!r}; expected one of {sorted(valid_answers)}.')
        return target

    def _resolve_video(self, record: Dict[str, Any]) -> str:
        if self.video_source == VIDEO_SOURCE_URL:
            return str(record['url'])
        if self.video_source == VIDEO_SOURCE_ARCHIVE:
            return self._ensure_video_file(str(record['video_id']))
        raise ValueError(f'Unsupported Video-MME-v2 video source: {self.video_source}')

    def _ensure_video_file(self, video_id: str) -> str:
        output_path = self._cache_output_path(video_id)
        if os.path.exists(output_path) and not self.force_redownload:
            return output_path

        archive_path = self.source_dataset.download_file(f'videos/{self._archive_name(video_id)}')
        member_name = self._find_archive_member(archive_path, video_id)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with zipfile.ZipFile(archive_path) as zip_file:
            with zip_file.open(member_name) as source, open(output_path, 'wb') as target:
                target.write(source.read())
        return output_path

    def _cache_output_path(self, video_id: str) -> str:
        video_id = self._normalize_video_id(video_id)
        output_path = os.path.abspath(os.path.join(self.video_cache_dir, f'{video_id}.mp4'))
        cache_dir = os.path.abspath(self.video_cache_dir)
        if os.path.commonpath([cache_dir, output_path]) != cache_dir:
            raise ValueError(f'Invalid Video-MME-v2 video id: {video_id}')
        return output_path

    @classmethod
    def _archive_name(cls, video_id: str) -> str:
        video_num = int(cls._normalize_video_id(video_id))
        archive_num = ((video_num - 1) // 20) + 1
        return f'{archive_num:03d}.zip'

    @staticmethod
    def _normalize_video_id(video_id: str) -> str:
        if not re.fullmatch(r'\d{1,3}', str(video_id)):
            raise ValueError(f'Invalid Video-MME-v2 video id: {video_id}')
        video_num = int(video_id)
        if not 1 <= video_num <= 800:
            raise ValueError(f'Invalid Video-MME-v2 video id: {video_id}')
        return f'{video_num:03d}'

    @classmethod
    def _find_archive_member(cls, archive_path: str, video_id: str) -> str:
        expected = f'{cls._normalize_video_id(video_id)}.mp4'
        with zipfile.ZipFile(archive_path) as zip_file:
            matches = [
                name for name in zip_file.namelist()
                if not name.endswith('/') and (name.endswith(f'/{expected}') or name.endswith(expected))
            ]
        if not matches:
            raise FileNotFoundError(f'Video {expected} was not found in archive {archive_path}.')
        return sorted(matches)[0]

    def _subtitle_for_record(self, video_id: str) -> str:
        video_id = self._normalize_video_id(video_id)
        if video_id not in self._subtitle_cache:
            subtitle_path = self.source_dataset.download_file('subtitle.zip')
            member_name = f'subtitle/{video_id}.jsonl'
            with zipfile.ZipFile(subtitle_path) as zip_file:
                try:
                    raw_text = zip_file.read(member_name).decode('utf-8')
                except KeyError as exc:
                    raise FileNotFoundError(f'Subtitle {member_name} was not found in {subtitle_path}.') from exc
            self._subtitle_cache[video_id] = self._subtitle_text_from_jsonl(raw_text)
        return self._subtitle_cache[video_id]

    def _subtitle_text_from_jsonl(self, raw_text: str) -> str:
        words = []
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get('text') or '').strip()
            if text:
                words.append(text)
                if self.subtitle_word_limit and len(words) >= self.subtitle_word_limit:
                    break
        return ' '.join(words)

    def _question_with_context(self, record: Dict[str, Any], subtitle: Optional[str]) -> str:
        question = str(record['question'])
        if not subtitle:
            return question
        return f'Subtitles:\n{subtitle}\n\n{question}'
