import contextlib
import copy
import io
import json
import os
import random
import re
import shutil
import string
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evalscope.api.benchmark import VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText, ContentVideo
from evalscope.api.metric import SampleScore, Score
from evalscope.constants import HubType
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import csv_to_list, jsonl_to_list, tsv_to_list
from evalscope.utils.logger import get_logger
from evalscope.utils.url_utils import guess_video_format, is_data_uri, is_http_url

logger = get_logger()

CAPTION_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
CAPTION_MAIN_SCORE = 'CIDEr'
DEFAULT_CAPTION_PROMPT = 'Describe the video in one concise sentence.'

ANSWER_LINE_PATTERN = re.compile(r'ANSWER:\s*(.*)', flags=re.IGNORECASE)
ARTICLES = {'a', 'an', 'the'}
NUMBER_MAP = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
}


def unique_texts(values: Iterable[str]) -> List[str]:
    """Return non-empty strings in stable unique order."""
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def ensure_text_list(value: Any) -> List[str]:
    """Coerce common dataset answer/caption shapes to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        text = value.decode('utf-8', errors='ignore').strip()
        return [text] if text else []
    if isinstance(value, dict):
        for key in ('caption', 'answer', 'text', 'value'):
            if key in value:
                return ensure_text_list(value[key])
        return []
    if isinstance(value, Iterable):
        texts = []
        for item in value:
            texts.extend(ensure_text_list(item))
        return [text for text in texts if str(text).strip()]
    return [str(value)]


def answer_from_generation(prediction: str) -> str:
    """Extract a final answer line when the prompt requested ANSWER: output."""
    matches = ANSWER_LINE_PATTERN.findall(prediction or '')
    if matches:
        return matches[-1].strip().strip('"\'')
    return (prediction or '').strip().strip('"\'')


def normalize_vqa_answer(answer: str) -> str:
    """Normalize an answer with the standard VQA-style text rules."""
    text = answer.lower().strip()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = _process_punctuation(text)
    words = []
    for word in text.split():
        mapped = NUMBER_MAP.get(word, word)
        if mapped not in ARTICLES:
            words.append(mapped)
    return ' '.join(words)


def vqa_soft_accuracy(prediction: str, answers: List[str]) -> float:
    """Compute VQAv2 soft accuracy against 10 human answers."""
    normalized_prediction = normalize_vqa_answer(prediction)
    normalized_answers = [normalize_vqa_answer(answer) for answer in answers]
    if not normalized_prediction or not normalized_answers:
        return 0.0

    matching_count = sum(answer == normalized_prediction for answer in normalized_answers)
    return min(1.0, matching_count / 3.0)


def exact_match(prediction: str, references: List[str]) -> float:
    """Return 1.0 when the normalized prediction matches any reference."""
    normalized_prediction = normalize_vqa_answer(prediction)
    normalized_references = {normalize_vqa_answer(reference) for reference in references}
    return float(normalized_prediction in normalized_references)


def compute_caption_scores(predictions: List[str], references: List[List[str]]) -> List[Dict[str, float]]:
    """Compute COCO-style caption metrics for a batch of predictions."""
    check_import(
        module_name='pycocoevalcap',
        extra='caption',
        feature_name='caption benchmark metrics',
        raise_error=True,
    )
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    gts = {}
    res = {}
    for index, (prediction, sample_references) in enumerate(zip(predictions, references)):
        gts[index] = [{'caption': reference} for reference in sample_references]
        res[index] = [{'caption': prediction}]

    use_official_java = shutil.which('java') is not None
    if use_official_java:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
    else:
        logger.warning(
            'Java is not available; caption metrics will use a pure-Python tokenizer and METEOR fallback. '
            'Install Java to reproduce the official COCO caption tokenizer and METEOR scores.'
        )
        gts = _simple_caption_tokenize(gts)
        res = _simple_caption_tokenize(res)

    results = [{metric: 0.0 for metric in CAPTION_METRICS} for _ in predictions]
    with contextlib.redirect_stdout(io.StringIO()):
        _, bleu_scores = Bleu(4).compute_score(gts, res)
    for metric_index, metric_name in enumerate(['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']):
        for sample_index, sample_score in enumerate(bleu_scores[metric_index]):
            results[sample_index][metric_name] = float(sample_score)

    for metric_name, scorer in [('ROUGE_L', Rouge()), ('CIDEr', Cider())]:
        with contextlib.redirect_stdout(io.StringIO()):
            _, sample_scores = scorer.compute_score(gts, res)
        for sample_index, sample_score in enumerate(sample_scores):
            results[sample_index][metric_name] = float(sample_score)

    if use_official_java:
        from pycocoevalcap.meteor.meteor import Meteor

        with contextlib.redirect_stdout(io.StringIO()):
            _, sample_scores = Meteor().compute_score(gts, res)
        for sample_index, sample_score in enumerate(sample_scores):
            results[sample_index]['METEOR'] = float(sample_score)
    else:
        for sample_index, (prediction, sample_references) in enumerate(zip(predictions, references)):
            results[sample_index]['METEOR'] = _simple_meteor_score(prediction, sample_references)

    return results


def _process_punctuation(text: str) -> str:
    punctuation = string.punctuation.replace(':', '')
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    return ''.join(' ' if char in punctuation else char for char in text)


def _caption_tokens(text: str) -> List[str]:
    text = _process_punctuation(text.lower())
    return [token for token in text.split() if token]


def _simple_caption_tokenize(data: Dict[int, List[Dict[str, str]]]) -> Dict[int, List[str]]:
    tokenized = {}
    for image_id, annotations in data.items():
        tokenized[image_id] = [' '.join(_caption_tokens(annotation['caption'])) for annotation in annotations]
    return tokenized


def _simple_meteor_score(prediction: str, references: List[str]) -> float:
    prediction_tokens = _caption_tokens(prediction)
    if not prediction_tokens:
        return 0.0

    best_score = 0.0
    prediction_counts = Counter(prediction_tokens)
    for reference in references:
        reference_tokens = _caption_tokens(reference)
        if not reference_tokens:
            continue
        reference_counts = Counter(reference_tokens)
        matches = sum(min(count, reference_counts[token]) for token, count in prediction_counts.items())
        if matches == 0:
            continue
        precision = matches / len(prediction_tokens)
        recall = matches / len(reference_tokens)
        score = (10 * precision * recall) / (recall + 9 * precision)
        best_score = max(best_score, score)
    return best_score


class CaptionDatasetAdapter(VisionLanguageAdapter):
    """Shared loading and media helpers for caption-style multimodal generation tasks."""

    media_type: str = 'video'
    media_fields: List[str] = ['video', 'video_path', 'url']
    reference_fields: List[str] = ['references', 'caption', 'captions', 'answer', 'answers']
    group_key_field: Optional[str] = None
    source_dataset_ids: Dict[str, str] = {}
    source_eval_splits: Dict[str, str] = {}
    source_subset_names: Dict[str, Dict[str, Optional[str]]] = {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_batch_scoring = True
        self.add_aggregation_name = False

    @property
    def source_dataset_id(self) -> str:
        if self.dataset_id == self.name or self.dataset_id in self.source_dataset_ids.values():
            return self.source_dataset_ids.get(self.source_dataset_hub, self.dataset_id)
        return self.dataset_id

    @property
    def source_dataset_hub(self) -> str:
        if os.path.exists(self.dataset_id):
            return HubType.LOCAL
        return self.extra_params.get('dataset_hub') or HubType.MODELSCOPE

    @property
    def source_eval_split(self) -> str:
        return self.extra_params.get('eval_split'
                                     ) or self.source_eval_splits.get(self.source_dataset_hub, self.eval_split)

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
            with self._temporary_attribute('current_subset_name', subset):
                records = self._prepare_records(self._load_records(subset))
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
        references = self._references_from_record(record)
        if not references:
            raise ValueError(f'No references found for {self.name} record: {record}')

        content_list: List[Content] = [ContentText(text=self._prompt_from_record(record))]
        media_content, media_metadata = self._content_from_record(record)
        if media_content is not None:
            content_list.append(media_content)

        metadata = {
            'references': references,
            'subset': self.current_subset_name,
            'dataset_id': self.source_dataset_id,
            'dataset_hub': self.source_dataset_hub,
        }
        metadata.update(media_metadata)
        metadata.update(self._metadata_from_record(record))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(references, ensure_ascii=False),
            metadata=metadata,
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
        predictions = [self.filter_prediction(task_state.output.completion, task_state) for task_state in task_states]
        references = [self._references_from_task_state(task_state) for task_state in task_states]
        batch_scores = compute_caption_scores(predictions, references)
        for sample_score, values in zip(sample_scores, batch_scores):
            sample_score.score.value.update(values)
            sample_score.score.main_score_name = CAPTION_MAIN_SCORE
        return sample_scores

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return answer_from_generation(prediction)

    def _load_records(self, subset: str) -> List[Dict[str, Any]]:
        source_subset = self._source_subset_name(subset)
        if self.source_dataset_hub == HubType.LOCAL:
            return self._load_local_records(source_subset)
        logger.info(
            f'Loading {self.name} records from {self.source_dataset_hub}: '
            f'{self.source_dataset_id}, subset={source_subset}, split={self.source_eval_split}.'
        )
        dataset = self.source_dataset.load(split=self.source_eval_split, subset=source_subset)
        return list(dataset)

    def _source_subset_name(self, subset: str) -> Optional[str]:
        return self.source_subset_names.get(self.source_dataset_hub, {}).get(subset, subset)

    def _load_local_records(self, subset: Optional[str]) -> List[Dict[str, Any]]:
        path = self.source_dataset_id
        supported_loaders = {
            '.jsonl': jsonl_to_list,
            '.csv': csv_to_list,
            '.tsv': tsv_to_list,
        }
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext not in supported_loaders:
                raise FileNotFoundError(f'Unsupported local dataset file format for {path}.')
            return supported_loaders[ext](path)

        for ext, loader in supported_loaders.items():
            if subset:
                candidates = [f'{subset}_{self.source_eval_split}{ext}', f'{subset}{ext}']
            else:
                candidates = [f'{self.source_eval_split}{ext}', f'data{ext}']
            for candidate in candidates:
                file_path = os.path.join(path, candidate)
                if os.path.exists(file_path):
                    return loader(file_path)
        raise FileNotFoundError(f'No local data file found for subset={subset}, split={self.eval_split} in {path}.')

    def _prepare_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.group_key_field:
            return records

        grouped: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        for record in records:
            group_key = str(record.get(self.group_key_field) or record.get('id') or len(grouped))
            references = self._references_from_record(record)
            if group_key not in grouped:
                grouped[group_key] = copy.deepcopy(record)
                grouped[group_key]['references'] = []
            grouped[group_key]['references'].extend(references)

        for record in grouped.values():
            record['references'] = unique_texts(record['references'])
        return list(grouped.values())

    def _apply_limit(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.limit is None:
            return records
        if isinstance(self.limit, float):
            limit = int(len(records) * self.limit)
        else:
            limit = self.limit
        return records[:limit]

    def _references_from_record(self, record: Dict[str, Any]) -> List[str]:
        for field in self.reference_fields:
            if field in record:
                references = ensure_text_list(record.get(field))
                if references:
                    return unique_texts(references)
        return []

    def _references_from_task_state(self, task_state: TaskState) -> List[str]:
        references = ensure_text_list(task_state.metadata.get('references'))
        if references:
            return references
        try:
            return ensure_text_list(json.loads(task_state.target))
        except json.JSONDecodeError:
            return ensure_text_list(task_state.target)

    def _prompt_from_record(self, record: Dict[str, Any]) -> str:
        return self.prompt_template or DEFAULT_CAPTION_PROMPT

    def _metadata_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': record.get('id'),
            'video_id': record.get('video_id'),
            'image_id': record.get('image_id'),
            'question_id': record.get('question_id'),
            'source': record.get('source'),
            'category': record.get('category'),
        }

    def _content_from_record(self, record: Dict[str, Any]) -> Tuple[Optional[Content], Dict[str, Any]]:
        if self.media_type == 'image':
            return self._image_content_from_record(record)
        if self.media_type == 'video':
            return self._video_content_from_record(record)
        raise ValueError(f'Unsupported caption media type: {self.media_type}')

    def _image_content_from_record(self, record: Dict[str, Any]) -> Tuple[Optional[Content], Dict[str, Any]]:
        media_value, resolved = self._best_media_value(record, media_dir=self.extra_params.get('image_dir') or '')
        if media_value is None:
            return None, {'media_resolved': False}
        if isinstance(media_value, bytes):
            image = self._image_bytes_to_base64(media_value, default_format='jpeg')
        else:
            image = str(media_value)
        metadata = {'media_resolved': resolved}
        if not is_data_uri(image):
            metadata['image'] = image
        return ContentImage(image=image), metadata

    def _video_content_from_record(self, record: Dict[str, Any]) -> Tuple[Optional[Content], Dict[str, Any]]:
        media_value, resolved = self._best_media_value(record, media_dir=self.extra_params.get('video_dir') or '')
        if media_value is None:
            return None, {'media_resolved': False}
        video = str(media_value)
        start = self._optional_float(self._first_record_value(record, ['start', 'start time']), 'start')
        end = self._optional_float(self._first_record_value(record, ['end', 'end time']), 'end')
        fps = self._optional_float(record.get('fps'), 'fps')
        return (
            ContentVideo(video=video, format=guess_video_format(video), start=start, end=end, fps=fps),
            {
                'video': video,
                'start': start,
                'end': end,
                'fps': fps,
                'media_resolved': resolved
            },
        )

    def _best_media_value(self, record: Dict[str, Any], media_dir: str) -> Tuple[Optional[Any], bool]:
        fallback: Tuple[Optional[Any], bool] = (None, False)
        for field in self.media_fields:
            if field not in record:
                continue
            media_value, resolved = self._resolve_media_value(record[field], media_dir=media_dir)
            if media_value is None:
                continue
            if resolved:
                return media_value, True
            if fallback[0] is None:
                fallback = (media_value, False)
        return fallback

    def _resolve_media_value(self, value: Any, media_dir: str) -> Tuple[Optional[Any], bool]:
        if value is None:
            return None, False
        if isinstance(value, dict):
            if value.get('bytes') is not None:
                return value['bytes'], True
            for key in ('path', 'url', 'image', 'video', 'data'):
                if value.get(key):
                    return self._resolve_media_value(value[key], media_dir=media_dir)
            return None, False
        if isinstance(value, bytes):
            return value, True

        media = str(value).strip()
        if not media:
            return None, False
        if is_http_url(media) or is_data_uri(media) or os.path.exists(media):
            return media, True
        if media_dir:
            resolved = self._resolve_media_path(media_dir, media)
            if os.path.exists(resolved):
                return resolved, True
        return media, False

    def _resolve_media_path(self, media_dir: str, media: str) -> str:
        extension = self.extra_params.get('video_extension') or self.extra_params.get('image_extension') or ''
        if extension:
            extension = extension if extension.startswith('.') else f'.{extension}'
            media = f'{os.path.splitext(media)[0]}{extension}'
        media_dir_abs = os.path.abspath(os.path.expanduser(media_dir))
        resolved = os.path.abspath(os.path.join(media_dir_abs, media))
        if os.path.commonpath([media_dir_abs, resolved]) != media_dir_abs:
            raise ValueError(f'Invalid media path: {media}')
        return resolved

    @staticmethod
    def _first_record_value(record: Dict[str, Any], fields: List[str]) -> Any:
        for field in fields:
            if field in record and record[field] is not None:
                return record[field]
        return None

    @staticmethod
    def _optional_float(value: Any, field_name: str) -> Optional[float]:
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid {field_name} value: {value!r}') from exc


class VQACaptionAdapter(CaptionDatasetAdapter):
    """Shared adapter for open-ended VQA datasets scored with VQAv2 soft accuracy."""

    media_type = 'image'
    media_fields = ['image', 'image_path', 'image_url', 'url']
    reference_fields = ['answers', 'answer', 'multiple_choice_answer']

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_batch_scoring = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = str(record.get('question') or '').strip()
        answers = self._references_from_record(record)
        if not answers:
            raise ValueError(f'No VQA answers found for record: {record}')

        content_list: List[Content] = [ContentText(text=self._prompt_from_record(record))]
        media_content, media_metadata = self._content_from_record(record)
        if media_content is not None:
            content_list.append(media_content)

        metadata = {
            'question': question,
            'answers': answers,
            'references': answers,
            'multiple_choice_answer': record.get('multiple_choice_answer'),
            'question_id': record.get('question_id'),
            'question_type': record.get('question_type'),
            'answer_type': record.get('answer_type'),
            'subset': self.current_subset_name,
            'dataset_id': self.source_dataset_id,
            'dataset_hub': self.source_dataset_hub,
        }
        metadata.update(media_metadata)

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(answers, ensure_ascii=False),
            metadata=metadata,
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        answers = ensure_text_list(task_state.metadata.get('answers'))
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value = {
            'vqa_score': vqa_soft_accuracy(filtered_prediction, answers),
            'exact_match': exact_match(filtered_prediction, answers),
        }
        score.main_score_name = 'vqa_score'
        return score

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return answer_from_generation(prediction)

    def _prompt_from_record(self, record: Dict[str, Any]) -> str:
        question = str(record.get('question') or '').strip()
        return self.prompt_template.format(question=question)

    def _references_from_record(self, record: Dict[str, Any]) -> List[str]:
        answers = ensure_text_list(record.get('answers'))
        if answers:
            return answers
        return ensure_text_list(record.get('multiple_choice_answer') or record.get('answer'))
