# flake8: noqa: E501
import os
import re
from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import (
    DatasetDict,
    Sample,
    build_dataset_dict_from_record_map,
    resolve_snapshot_or_local_path,
)
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

# Track (level 1) -> sub-task (level 2). Sub-tasks are the report subsets; the finer-grained
# scenario directories (level 3) only drive scenario-specific scoring rules.
TASK_STRUCTURE: Dict[str, List[str]] = {
    'recognition': ['multi_lingual_recognition', 'natural_scene_recognition'],
    'parsing': [
        'complex_table_parsing', 'formula_parsing', 'general_documents_parsing', 'info_board_parsing',
        'molecular_parsing'
    ],
    'grounding': ['object_grounding', 'text_grounding'],
    'extraction': ['business_transactions', 'public_services', 'regulated_records'],
    'qa': ['blueprint_qa', 'dashboards_fact_qa', 'dashboards_numeric_qa', 'financial_documents_qa'],
}

SUBSET_TO_TRACK: Dict[str, str] = {
    sub_task: track
    for track, sub_tasks in TASK_STRUCTURE.items()
    for sub_task in sub_tasks
}

SUBSET_LIST: List[str] = list(SUBSET_TO_TRACK)

DESCRIPTION = """
## Overview

CC-OCR V2 is a challenging OCR benchmark tailored to real-world enterprise document processing. It deliberately
over-samples the hard and corner cases that prior OCR benchmarks under-represent, such as photographed and
scanned tables, handwritten formulas, multi-page receipts, and low-quality multilingual scene text.

## Task Description

- **Task Type**: Text recognition, document parsing, document grounding, key information extraction, and document VQA
- **Input**: One or more document images plus the task instruction shipped with each sample
- **Output**: Free-form text, LaTeX, HTML tables, SMILES strings, JSON objects, or bounding boxes, depending on the track
- **Modalities**: Image + text, bilingual (Chinese / English) with 32 additional languages in the recognition track

## Key Features

- 7,093 official samples over 5 tracks and 16 sub-tasks, evaluated as one benchmark; 7,091 are
  loaded because the dataset repository ships no image for two of them
- **recognition**: multilingual (32 languages) and natural-scene text reading
- **parsing**: complex tables, general documents, handwritten formulas, molecular structures, and information boards
- **grounding**: text grounding (single box) and object grounding (multi-box detection with labels)
- **extraction**: schema-driven key information extraction over business, public-service, and regulated records
- **qa**: question answering over blueprints, dashboards, and financial documents
- Prompts come from the official dataset, so results stay comparable to the published leaderboard

## Evaluation Notes

- Every sample yields one `score` in `[0, 1]`; each track uses its official metric:
  recognition = token-level F1, parsing = edit similarity / TEDS, grounding = IoU,
  extraction = field-level F1, qa = substring match with ANLS fallback
- Subset scores are sample means; the per-track category also reports a macro average over its sub-tasks
- The grounding prompts ask for boxes on a 0-1000 grid; predictions in absolute pixels are rescaled
  as if normalized and therefore score close to zero, matching the official leaderboard behavior
- Full-page parsing targets are long, so allow a generous `max_tokens` (4096 or more)
- Requires: `apted`, `distance`, `lxml`, `python-Levenshtein`, `scipy`, `zss`
  (`pip install 'evalscope[cc_ocr_v2]'`)
- The dataset is a file tree of images and answers (about 5 GB). Only the tracks listed in
  `subset_list` are downloaded, so restricting subsets keeps the download small
"""


@register_benchmark(
    BenchmarkMeta(
        name='cc_ocr_v2',
        pretty_name='CC-OCR-V2',
        tags=[Tags.MULTI_MODAL, Tags.QA, Tags.GROUNDING, Tags.MULTI_LINGUAL],
        description=DESCRIPTION,
        dataset_id='evalscope/CC-OCR-V2',
        subset_list=SUBSET_LIST,
        metric_list=['score'],
        eval_split='test',
        paper_url='https://arxiv.org/abs/2605.03903',
    )
)
class CCOCRV2Adapter(VisionLanguageAdapter):
    """CC-OCR V2 is distributed as a file tree of ``question`` / ``answer`` / ``images`` folders,
    so the dataset is assembled by scanning the snapshot instead of loading a tabular split."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_map = SUBSET_TO_TRACK

    def load(self) -> Tuple[DatasetDict, None]:
        check_import(
            module_name=['apted', 'distance', 'Levenshtein', 'lxml', 'scipy', 'zss'],
            extra='cc_ocr_v2',
            raise_error=True,
            feature_name='CC-OCR-V2 benchmark',
        )

        unknown = [subset for subset in self.subset_list if subset not in SUBSET_TO_TRACK]
        if unknown:
            raise ValueError(f'Unknown CC-OCR V2 subsets {unknown}. Valid subsets are: {SUBSET_LIST}')

        # Only fetch the selected sub-tasks; the full benchmark is roughly 5 GB of images.
        dataset_root = resolve_snapshot_or_local_path(
            self, allow_file_pattern=[f'{SUBSET_TO_TRACK[subset]}/{subset}/*' for subset in self.subset_list]
        )

        record_map = {}
        for subset in self.subset_list:
            records = self._scan_sub_task(dataset_root, subset)
            if records:
                record_map[subset] = records

        if not record_map:
            raise FileNotFoundError(f'No CC-OCR V2 samples found under {dataset_root}.')

        return build_dataset_dict_from_record_map(
            record_map=record_map,
            sample_fields=self.record_to_sample,
            location=self.dataset_id,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=None,
        ), None

    def _scan_sub_task(self, dataset_root: str, sub_task: str) -> List[Dict[str, Any]]:
        """Collect one record per question file, pairing it with its answer and image(s)."""
        task_dir = os.path.join(dataset_root, SUBSET_TO_TRACK[sub_task], sub_task)
        question_root = os.path.join(task_dir, 'question')
        if not os.path.isdir(question_root):
            logger.warning(f'CC-OCR V2 sub-task directory is missing: {question_root}')
            return []

        records: List[Dict[str, Any]] = []
        for scenario in sorted(os.listdir(question_root)):
            scenario_dir = os.path.join(question_root, scenario)
            if not os.path.isdir(scenario_dir):
                continue
            image_index = _index_images(os.path.join(task_dir, 'images', scenario))
            for question_name in sorted(os.listdir(scenario_dir)):
                if not question_name.endswith('.txt'):
                    continue
                sample_id = question_name[:-len('.txt')]
                answer_path = os.path.join(task_dir, 'answer', scenario, question_name)
                image_paths = image_index.get(sample_id, [])
                if not os.path.isfile(answer_path) or not image_paths:
                    # Samples without an answer or image cannot be scored; drop them loudly
                    # rather than letting them reach the model and score zero.
                    logger.warning(f'Skipping incomplete CC-OCR V2 sample {sub_task}/{scenario}/{sample_id}.')
                    continue
                records.append({
                    'id': sample_id,
                    'sub_task': sub_task,
                    'scenario': scenario,
                    'question': _read_text(os.path.join(scenario_dir, question_name)),
                    'answer': _read_text(answer_path),
                    'image_paths': image_paths,
                })
        return records

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Images precede the instruction, matching the official request builder.
        content: List[Content] = [ContentImage(image=path) for path in record['image_paths']]
        content.append(ContentText(text=record['question']))

        sub_task = record['sub_task']
        return Sample(
            input=[ChatMessageUser(content=content)],
            target=record['answer'],
            subset_key=sub_task,
            metadata={
                'id': record['id'],
                'task': SUBSET_TO_TRACK[sub_task],
                'sub_task': sub_task,
                'scenario': record['scenario'],
                # Grounding scoring reads the image size from this path to map predicted
                # coordinates into pixels, so it must survive into the review records.
                'image_paths': record['image_paths'],
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from .utils import score_sample

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        score.value = {'score': score_sample(filtered_prediction, reference, task_state.metadata or {})}
        score.main_score_name = 'score'
        return score


def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as file:
        return file.read()


def _natural_key(name: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def _index_images(scenario_image_dir: str) -> Dict[str, List[str]]:
    """Map each sample id to its image(s): a single ``<id>.<ext>`` file, or the page files of a
    ``<id>/`` directory for multi-page documents."""
    index: Dict[str, List[str]] = {}
    if not os.path.isdir(scenario_image_dir):
        return index

    for name in os.listdir(scenario_image_dir):
        path = os.path.join(scenario_image_dir, name)
        if os.path.isdir(path):
            pages = [page for page in os.listdir(path) if os.path.splitext(page)[1].lower() in IMAGE_EXTENSIONS]
            if pages:
                index[name] = [os.path.join(path, page) for page in pages]
        else:
            stem, extension = os.path.splitext(name)
            if extension.lower() in IMAGE_EXTENSIONS:
                index.setdefault(stem, []).append(path)

    for paths in index.values():
        # Natural order so page_2 precedes page_10 in multi-page documents.
        paths.sort(key=lambda page_path: _natural_key(os.path.basename(page_path)))
    return index
