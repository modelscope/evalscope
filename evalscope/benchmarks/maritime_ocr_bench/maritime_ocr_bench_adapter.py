import base64
import io
import math
import mimetypes
import os
from PIL import Image
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, LocalDataLoader, Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

# Keep request payloads within service limits and avoid invalid extreme aspect ratios.
MAX_ASPECT_RATIO = 180.0
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_BYTES = 1_500_000
SAFE_RAW_IMAGE_BYTES = 1_000_000

SUBSET_LIST = ['IE', 'VQA', 'parsing', 'json1', 'json2']

TASK_TYPE_MAP = {
    # Official 1888-sample test split task types.
    'ie': 'IE',
    'vqa': 'VQA',
    'parsing': 'parsing',
    'json1': 'json1',
    'json2': 'json2'
}


@register_benchmark(
    BenchmarkMeta(
        name='maritime_ocr_bench',
        pretty_name='Maritime-OCR-Bench',
        tags=[Tags.MULTI_MODAL, Tags.QA],
        description="""
## Overview

Maritime-OCR-Bench is a comprehensive evaluation benchmark for assessing multimodal large model capabilities
on OCR-related tasks. The current released set contains 1,888 manually curated samples across five task types.

## Task Types

- **VQA**: Visual question answering on document/scene images
- **IE**: Information extraction requiring strict JSON output
- **parsing**: Text recognition and parsing from images
- **json1**: Text spotting with JSON v1 structured output
- **json2**: Text spotting with JSON v2 structured output

## Evaluation Metrics

Each task type uses a specialized scoring method:
- VQA/parsing: Multi-dimensional text similarity (edit distance, char F1, LCS F1, table-aware similarity)
- IE: Text coverage + JSON strictness (0.5 * coverage + 0.5 * json_strict)
- json1/json2: DIoU layout score + text score (0.7 * diou + 0.3 * text)
""",
        dataset_id='HiDolphin/MaritimeOCRBench',
        subset_list=SUBSET_LIST,
        metric_list=['score'],
        eval_split='test',
        prompt_template='{question}',
    )
)
class MaritimeOCRBenchAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.add_aggregation_name = False
        self.data_root: Optional[str] = None

    def load(self):
        """Load benchmark records from a local JSONL file or downloaded snapshot."""
        check_import(
            module_name='shapely',
            extra='maritime_ocr_bench',
            raise_error=True,
            feature_name='Maritime-OCR-Bench benchmark',
        )
        dataset_name_or_path = self.dataset_id

        if os.path.exists(dataset_name_or_path):
            dataset_path = dataset_name_or_path
            logger.info(f'Loading Maritime-OCR-Bench from local path: {dataset_path}')
        else:
            from modelscope import dataset_snapshot_download

            logger.info(f'Downloading Maritime-OCR-Bench dataset from ModelScope: {dataset_name_or_path}')
            dataset_path = dataset_snapshot_download(dataset_name_or_path)

        dataset_file = self._resolve_dataset_file(dataset_path)
        self.data_root = os.path.dirname(dataset_file)

        dataset = LocalDataLoader(
            data_id_or_path=dataset_file,
            split=self.eval_split,
            sample_fields=self.record_to_sample,
            subset='test',
            limit=None,
            repeats=self.repeats,
            shuffle=self.shuffle,
        ).load()

        return DatasetDict.from_dataset(
            dataset=dataset,
            subset_list=self.subset_list,
            limit=self.limit,
            repeats=self.repeats,
        ), None

    def _resolve_dataset_file(self, dataset_path: str) -> str:
        if os.path.isfile(dataset_path):
            return dataset_path

        preferred_files = ['maritime_ocr_bench.jsonl']
        for file_name in preferred_files:
            candidate = os.path.join(dataset_path, file_name)
            if os.path.exists(candidate):
                return candidate

        jsonl_files: List[str] = []
        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                if file_name.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file_name))

        if not jsonl_files:
            raise FileNotFoundError(f'No JSONL dataset file found under: {dataset_path}')

        return sorted(jsonl_files)[0]

    def _resolve_image_path(self, image_ref: str) -> str:
        normalized_ref = os.path.normpath(image_ref)

        if any(part == os.pardir for part in normalized_ref.split(os.sep)):
            raise ValueError(f'Parent directory references are not allowed in image paths: {image_ref}')

        if os.path.isabs(normalized_ref) and os.path.exists(normalized_ref):
            return normalized_ref

        candidate_paths: List[str] = []
        if self.data_root:
            candidate_paths.append(os.path.join(self.data_root, normalized_ref))

            path_parts = normalized_ref.split(os.sep, 1)
            if len(path_parts) == 2:
                candidate_paths.append(os.path.join(self.data_root, path_parts[1]))

            candidate_paths.append(os.path.join(self.data_root, 'images', os.path.basename(normalized_ref)))

        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return candidate

        return candidate_paths[0] if candidate_paths else normalized_ref

    def _read_raw_image_data_url(self, image_path: str) -> tuple[str, str]:
        mime, _ = mimetypes.guess_type(image_path)
        mime = mime or 'image/jpeg'
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        return f'data:{mime};base64,{b64}', mime

    def _should_normalize_image(self, image_path: str) -> bool:
        file_size = os.path.getsize(image_path)
        if file_size > SAFE_RAW_IMAGE_BYTES:
            return True

        with Image.open(image_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return True
            if max(width, height) > MAX_IMAGE_SIDE:
                return True
            if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
                return True

        return False

    def _normalize_image_bytes(self, image_path: str) -> tuple[str, str]:
        """Return image as normalized data URL and MIME type for robust API serving."""
        with Image.open(image_path) as img:
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            width, height = img.size
            if width <= 0 or height <= 0:
                raise ValueError(f'Invalid image size for {image_path}: {img.size}')

            # First, bound the long edge.
            scale = min(1.0, float(MAX_IMAGE_SIDE) / float(max(width, height)))
            target_w = max(1, int(width * scale))
            target_h = max(1, int(height * scale))

            # Then, enforce aspect-ratio constraints required by the serving model.
            if target_w / target_h > MAX_ASPECT_RATIO:
                target_h = max(target_h, int(math.ceil(target_w / MAX_ASPECT_RATIO)))
            elif target_h / target_w > MAX_ASPECT_RATIO:
                target_w = max(target_w, int(math.ceil(target_h / MAX_ASPECT_RATIO)))

            if (target_w, target_h) != img.size:
                img = img.resize((target_w, target_h), Image.Resampling.BICUBIC)

            # Encode to JPEG and reduce quality until size is acceptable.
            quality = 90
            encoded: bytes = b''
            while quality >= 45:
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=quality, optimize=True)
                encoded = buf.getvalue()
                if len(encoded) <= MAX_IMAGE_BYTES:
                    break
                quality -= 10

            b64 = base64.b64encode(encoded).decode('ascii')
            return f'data:image/jpeg;base64,{b64}', 'image/jpeg'

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        messages = record.get('messages', [])
        images = record.get('images', [])
        task_type = record.get('task_type', 'VQA')

        # Extract prompt and solution
        prompt = messages[0].get('content', '') if messages else ''
        solution = messages[1].get('content', '') if len(messages) > 1 else ''

        # Clean <image> placeholder from prompt
        cleaned_prompt = prompt.replace('<image>', '').strip()

        # Build multimodal content list
        content_list: List[Content] = [ContentText(text=cleaned_prompt)]

        # Load image and encode as base64
        if images:
            image_path = self._resolve_image_path(images[0])
            if os.path.exists(image_path):
                try:
                    if self._should_normalize_image(image_path):
                        image_data_url, _ = self._normalize_image_bytes(image_path)
                    else:
                        image_data_url, _ = self._read_raw_image_data_url(image_path)
                    content_list.append(ContentImage(image=image_data_url))
                except Exception as e:
                    logger.warning(f'Image normalize failed for {image_path}, fallback raw bytes: {e}')
                    image_data_url, _ = self._read_raw_image_data_url(image_path)
                    content_list.append(ContentImage(image=image_data_url))
            else:
                logger.warning(f'Image not found: {image_path}')

        # Normalize task type
        normalized_type = TASK_TYPE_MAP.get(str(task_type).strip().lower())
        if normalized_type is None:
            raise ValueError(f'Unsupported maritime_ocr_bench task_type: {task_type}')

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=solution,
            subset_key=normalized_type,
            metadata={
                'task_type': normalized_type,
                'prompt': prompt,
                'images': images,
            }
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        task_type = task_state.metadata.get('task_type', 'VQA')
        prompt = task_state.metadata.get('prompt', '')
        completion = filtered_prediction
        solution = reference

        # Lazy import scoring modules so optional dependencies (e.g. shapely)
        # are only resolved when this benchmark is actually executed.
        from .scoring.IE_eval import evaluate_ie_sample
        from .scoring.spotting_json_quantization_eval import evaluate_spotting_sample
        from .scoring.spotting_quantization_eval import evaluate_text_sample

        if task_type == 'json1':
            metrics = evaluate_spotting_sample(
                completion=completion,
                solution=solution,
                expected_format='v1',
            )
            score_value = float(metrics['overall_score'])

        elif task_type == 'json2':
            metrics = evaluate_spotting_sample(
                completion=completion,
                solution=solution,
                expected_format='v2',
            )
            score_value = float(metrics['overall_score'])

        elif task_type == 'IE':
            metrics = evaluate_ie_sample(
                completion=completion,
                solution=solution,
                prompt=prompt,
            )
            score_value = float(metrics['ie_score'])

        else:
            # VQA, parsing, and fallback
            metrics = evaluate_text_sample(
                completion=completion,
                solution=solution,
            )
            score_value = float(metrics['text_score'])

        score.value = {'score': score_value}
        return score
