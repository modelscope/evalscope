import copy
import json
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentText, ContentVideo
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.caption.metrics import CAPTION_MAIN_SCORE, CAPTION_METRICS, compute_caption_scores
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.url_utils import guess_video_format

logger = get_logger()

DEFAULT_PROMPT = 'Describe the video in one concise sentence.'


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

- Default data source: `evalscope/MSVD` on ModelScope, `test` split
- Hugging Face `VLM2Vec/MSVD` remains available by setting `extra_params.dataset_hub="huggingface"`
- Primary metric: **CIDEr**
- Additional metrics: BLEU-1/2/3/4, METEOR, ROUGE-L
- Set `extra_params.video_dir` when the dataset only provides video file names and local media files are required
""",
        tags=[Tags.MULTI_MODAL, Tags.IMAGE_CAPTIONING],
        dataset_id='evalscope/MSVD',
        paper_url='https://aclanthology.org/P11-1020/',
        subset_list=['default'],
        metric_list=CAPTION_METRICS,
        eval_split='test',
        prompt_template=DEFAULT_PROMPT,
        extra_params={
            'dataset_hub': {
                'type': 'str',
                'description': 'Dataset hub used to load MSVD annotations.',
                'value': HubType.MODELSCOPE,
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
class MSVDAdapter(VisionLanguageAdapter):
    """Adapter for MSVD video captioning."""

    SOURCE_DATASET_IDS = {
        HubType.MODELSCOPE: 'evalscope/MSVD',
        HubType.HUGGINGFACE: 'VLM2Vec/MSVD',
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_batch_scoring = True
        self.add_aggregation_name = False

    @property
    def source_dataset_hub(self) -> str:
        return self.extra_params.get('dataset_hub') or HubType.MODELSCOPE

    @property
    def source_dataset_id(self) -> str:
        if self.dataset_id != self.name and self.dataset_id not in self.SOURCE_DATASET_IDS.values():
            return self.dataset_id
        return self.SOURCE_DATASET_IDS.get(self.source_dataset_hub, self.dataset_id)

    @property
    def source_eval_split(self) -> str:
        return self.extra_params.get('eval_split') or self.eval_split

    @property
    def source_dataset(self) -> DatasetHub:
        return DatasetHub(
            data_id_or_path=self.source_dataset_id,
            data_source=self.source_dataset_hub,
            revision=self.extra_params.get('dataset_revision') or None,
            force_redownload=self.force_redownload,
        )

    def load_dataset(self) -> DatasetDict:
        dataset_dict: Dict[str, MemoryDataset] = {}
        for subset in self.subset_list:
            with self._temporary_attribute('current_subset_name', subset):
                records = self._group_records(self._load_records())
                if self.shuffle:
                    random.Random(self.seed).shuffle(records)
                records = self._apply_limit(records)
                samples = [self.record_to_sample(record) for record in records]
                if self.repeats > 1:
                    samples = [copy.deepcopy(sample) for sample in samples for _ in range(self.repeats)]
                dataset = MemoryDataset(samples=samples, name=self.name, location=self.source_dataset_id)
                dataset.reindex(group_size=self.repeats)
                dataset_dict[subset] = dataset

        self.test_dataset = DatasetDict(dataset_dict)
        self.fewshot_dataset = None
        self._post_process_samples()
        return self.test_dataset

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        references = record.get('references', [])
        if not references:
            raise ValueError(f'No references found for MSVD record: {record}')

        video = self._resolve_video(record)
        fps = _optional_float(record.get('fps'), 'fps')
        content_list: List[Content] = [ContentText(text=self.prompt_template or DEFAULT_PROMPT)]
        if video:
            content_list.append(ContentVideo(video=video, format=guess_video_format(video), fps=fps))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(references, ensure_ascii=False),
            metadata={
                'references': references,
                'subset': self.current_subset_name,
                'dataset_id': self.source_dataset_id,
                'dataset_hub': self.source_dataset_hub,
                'video': video,
                'video_id': record.get('video_id'),
                'source': record.get('source'),
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value = {metric: 0.0 for metric in CAPTION_METRICS}
        score.main_score_name = CAPTION_MAIN_SCORE
        return score

    def batch_calculate_metrics(self, task_states: List[TaskState],
                                sample_scores: List[SampleScore]) -> List[SampleScore]:
        predictions = [self.filter_prediction(ts.output.completion, ts) for ts in task_states]
        references = [ts.metadata.get('references') or json.loads(ts.target) for ts in task_states]
        batch_scores = compute_caption_scores(predictions, references)
        for sample_score, values in zip(sample_scores, batch_scores):
            sample_score.score.value.update(values)
            sample_score.score.main_score_name = CAPTION_MAIN_SCORE
        return sample_scores

    def _load_records(self) -> List[Dict[str, Any]]:
        logger.info(
            f'Loading MSVD from {self.source_dataset_hub}: '
            f'{self.source_dataset_id}, split={self.source_eval_split}.'
        )
        dataset = self.source_dataset.load(split=self.source_eval_split, subset='default')
        return list(dataset)

    def _group_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        for record in records:
            video_id = str(record.get('video_id') or record.get('id') or len(grouped))
            captions = _extract_captions(record)
            if video_id not in grouped:
                grouped[video_id] = copy.deepcopy(record)
                grouped[video_id]['references'] = []
            grouped[video_id]['references'].extend(captions)

        for record in grouped.values():
            record['references'] = _unique_texts(record['references'])
        return list(grouped.values())

    def _resolve_video(self, record: Dict[str, Any]) -> Optional[str]:
        video_name = record.get('video') or record.get('video_path')
        if not video_name:
            return None
        video_name = str(video_name)
        video_dir = self.extra_params.get('video_dir') or ''
        if video_dir:
            extension = self.extra_params.get('video_extension') or ''
            if extension:
                extension = extension if extension.startswith('.') else f'.{extension}'
                video_name = f'{os.path.splitext(video_name)[0]}{extension}'
            return os.path.join(os.path.abspath(video_dir), video_name)
        return video_name

    def _apply_limit(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.limit is None:
            return records
        if isinstance(self.limit, float):
            limit = int(len(records) * self.limit)
        else:
            limit = self.limit
        return records[:limit]


def _optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {field_name} value: {value!r}') from exc


def _extract_captions(record: Dict[str, Any]) -> List[str]:
    for field in ('references', 'caption', 'captions'):
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
    return []


def _unique_texts(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
