# flake8: noqa: E501
import json
import os
import shutil
import zipfile
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, Sample, build_dataset_from_records
from evalscope.api.messages import ChatMessageUser, Content, ContentText, ContentVideo
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, prompt
from evalscope.utils.url_utils import guess_video_format

logger = get_logger()

DATASET_ID = 'evalscope/TVBench'
SUBSET_LIST = [
    'action_antonym',
    'action_count',
    'action_localization',
    'action_sequence',
    'egocentric_sequence',
    'moving_direction',
    'object_count',
    'object_shuffle',
    'scene_transition',
    'unexpected_action',
]
DEFAULT_SUBSET_LIST = ['action_count']

DESCRIPTION = """
## Overview

TVBench is a temporal video understanding benchmark for evaluating whether multimodal models can reason over dynamic visual events rather than isolated frames. It covers a broad set of video reasoning skills, including action recognition, action counting, temporal localization, action order understanding, egocentric action sequencing, object counting, object shuffling, moving direction recognition, scene transition reasoning, and unexpected action detection.

The EvalScope native adapter reads the official per-task JSON annotations from the dataset repository and resolves the corresponding video archives on demand. This keeps the default smoke-test path lightweight while still supporting the full benchmark through `subset_list`.

## Task Description

- **Task Type**: Video multiple-choice question answering (MCQ)
- **Input**: A video clip or a time-bounded video segment, a natural-language question, and 2-4 answer candidates
- **Output**: A single answer option letter selected from the provided candidates
- **Default Subset**: `action_count`, selected because it is available as a standard MP4 archive in the public dataset repository
- **Supported Subsets**: `action_antonym`, `action_count`, `action_localization`, `action_sequence`, `egocentric_sequence`, `moving_direction`, `object_count`, `object_shuffle`, `scene_transition`, and `unexpected_action`

## Evaluation Notes

- Default evaluation uses 0-shot Chain-of-Thought multiple-choice prompting via `MultipleChoiceTemplate.SINGLE_ANSWER_COT`.
- Primary metric: Accuracy (`acc`). The dataset answer is stored as candidate text and is converted to the corresponding option letter before scoring.
- Some subsets provide `start`/`end` fields. The adapter passes these values to `ContentVideo` and also adds a concise segment instruction to the prompt.
- Video files are downloaded lazily from subset-specific archives. `egocentric_sequence` uses segmented archives under `video/egocentric_sequence/<prefix>.zip`.
- The `action_antonym` annotations reference AVI files. If the repository media archive is unavailable, configure `extra_params.video_dir` to a local directory containing the AVI files.
- The adapter supports local video layouts that are either flat (`video_dir/<video>`) or grouped by subset (`video_dir/<subset>/<video>`).
"""


@register_benchmark(
    BenchmarkMeta(
        name='tvbench',
        pretty_name='TVBench',
        dataset_id=DATASET_ID,
        tags=[Tags.MULTI_MODAL, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION,
        subset_list=DEFAULT_SUBSET_LIST,
        metric_list=['acc'],
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
        extra_params={
            'video_dir': {
                'type': 'str',
                'description': 'Optional local directory containing TVBench video files. It may be organized flat or by subset.',
                'value': '',
            },
        },
    )
)
class TVBenchAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    """Native adapter for the TVBench video multiple-choice benchmark."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._video_cache_dir: Optional[str] = None

    @property
    def video_cache_dir(self) -> str:
        if self._video_cache_dir is None:
            self._video_cache_dir = os.path.join(self.dataset_dir, 'tvbench', 'videos')
        return self._video_cache_dir

    @property
    def source_dataset(self) -> DatasetHub:
        return DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.dataset_hub,
            force_redownload=self.force_redownload,
        )

    @property
    def local_video_dir(self) -> str:
        return self.extra_params.get('video_dir') or ''

    def load_dataset(self) -> DatasetDict:
        dataset_dict = {}
        for subset in self.subset_list:
            if subset not in SUBSET_LIST:
                raise ValueError(f'Unsupported TVBench subset: {subset}. Supported subsets: {SUBSET_LIST}')
            with self._temporary_attribute('current_subset_name', subset):
                records = self._load_subset_records(subset)
                dataset = build_dataset_from_records(
                    records=records,
                    sample_fields=self.record_to_sample,
                    name='tvbench',
                    location=self.dataset_id,
                    limit=self.limit,
                    repeats=self.repeats,
                    shuffle=self.shuffle,
                    seed=self.seed,
                )
                dataset_dict[subset] = dataset

        self.test_dataset = DatasetDict(dataset_dict)
        self.fewshot_dataset = None
        self._post_process_samples()
        return self.test_dataset

    def _load_subset_records(self, subset: str) -> List[Dict[str, Any]]:
        logger.info(f'Loading TVBench subset {subset} from {self.dataset_hub}: {self.dataset_id}.')
        json_path = self.source_dataset.download_file(f'json/{subset}.json')
        with open(json_path, 'r', encoding='utf-8') as json_file:
            records = json.load(json_file)
        if not isinstance(records, list) or not records:
            raise ValueError(f'No records found for TVBench subset: {subset}')
        return records

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [str(choice) for choice in record['candidates']]
        answer = str(record['answer'])
        if answer not in choices:
            raise ValueError(f'Invalid TVBench answer {answer!r}; expected one of {choices!r}.')
        target = answer_character(choices.index(answer))

        start = self._optional_float(record.get('start'), 'start')
        end = self._optional_float(record.get('end'), 'end')
        fps = self._optional_float(record.get('fps'), 'fps')
        if start is not None and end is not None and start > end:
            raise ValueError(f'Invalid TVBench time boundary: start={start} is greater than end={end}.')

        video_name = str(record['video'])
        video_path = self._resolve_video(self.current_subset_name, video_name)
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
                'accurate_start': record.get('accurate_start'),
                'accurate_end': record.get('accurate_end'),
                'video_length': record.get('video_length'),
                'question_id': record.get('question_id'),
                'dataset_id': self.dataset_id,
                'dataset_hub': self.dataset_hub,
            },
        )

    def _resolve_video(self, subset: str, video_name: str) -> str:
        if self.local_video_dir:
            return self._local_video_path(self.local_video_dir, subset, video_name)
        return self._ensure_video_file(subset, video_name)

    def _local_video_path(self, video_dir: str, subset: str, video_name: str) -> str:
        root_dir = os.path.abspath(video_dir)
        candidate_paths = [
            self._safe_join(root_dir, subset, video_name),
            self._safe_join(root_dir, video_name),
        ]
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                return candidate_path
        raise FileNotFoundError(
            f'TVBench video {video_name!r} was not found under local video_dir {root_dir!r} '
            f'(checked subset and flat layouts).'
        )

    def _ensure_video_file(self, subset: str, video_name: str) -> str:
        output_path = self._cache_output_path(subset, video_name)
        if os.path.exists(output_path) and not self.force_redownload:
            return output_path

        archive_path = self.source_dataset.download_file(self._archive_path(subset, video_name))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with zipfile.ZipFile(archive_path) as zip_file:
            member_name = self._find_archive_member(zip_file, video_name)
            with zip_file.open(member_name) as source, open(output_path, 'wb') as target:
                shutil.copyfileobj(source, target)
        return output_path

    def _cache_output_path(self, subset: str, video_name: str) -> str:
        subset_dir = os.path.abspath(os.path.join(self.video_cache_dir, subset))
        return self._safe_join(subset_dir, video_name)

    @staticmethod
    def _safe_join(root_dir: str, *path_parts: str) -> str:
        output_path = os.path.abspath(os.path.join(root_dir, *path_parts))
        if os.path.commonpath([root_dir, output_path]) != root_dir:
            raise ValueError(f'Invalid TVBench video path: {os.path.join(*path_parts)}')
        return output_path

    @staticmethod
    def _archive_path(subset: str, video_name: str) -> str:
        if subset == 'action_antonym':
            raise FileNotFoundError(
                'The TVBench repository does not expose video/action_antonym.zip. '
                'Please configure extra_params.video_dir with local action_antonym AVI files.'
            )
        if subset == 'egocentric_sequence':
            archive_name = video_name.split('/', 1)[0]
            if not archive_name or archive_name == video_name:
                raise ValueError(f'Invalid TVBench egocentric video path: {video_name!r}')
            return f'video/egocentric_sequence/{archive_name}.zip'
        return f'video/{subset}.zip'

    @staticmethod
    def _find_archive_member(zip_file: zipfile.ZipFile, video_name: str) -> str:
        normalized_video_name = video_name.replace('\\', '/').lstrip('/')
        video_basename = os.path.basename(normalized_video_name)
        member_names = [name for name in zip_file.namelist() if not name.endswith('/')]
        exact_matches = [
            name for name in member_names if name.replace('\\', '/').endswith(f'/{normalized_video_name}')
            or name.replace('\\', '/') == normalized_video_name
        ]
        if exact_matches:
            return sorted(exact_matches)[0]

        basename_matches = [
            name for name in member_names if os.path.basename(name.replace('\\', '/')) == video_basename
        ]
        if basename_matches:
            return sorted(basename_matches)[0]
        raise FileNotFoundError(f'Video {video_name} was not found in archive.')

    def _question_with_video_context(
        self,
        record: Dict[str, Any],
        start: Optional[float],
        end: Optional[float],
    ) -> str:
        context = self._format_video_context(start=start, end=end)
        question = str(record['question'])
        if not context:
            return question
        return f'{context}\n\n{question}'

    @classmethod
    def _format_video_context(cls, start: Optional[float], end: Optional[float]) -> str:
        time_range = cls._format_time_range(start, end)
        if not time_range:
            return ''
        return f'Answer based on the video segment {time_range}.'

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
            raise ValueError(f'Invalid TVBench {field_name}: {value!r}') from exc
