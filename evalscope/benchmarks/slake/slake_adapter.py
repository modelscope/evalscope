# flake8: noqa: E501
import json
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import (
    DatasetDict,
    Sample,
    build_dataset_dict_from_record_map,
    resolve_snapshot_or_local_path,
)
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from .utils import normalize_answer, parse_answer

IMAGE_ARCHIVE = 'imgs.zip'
IMAGE_DIR_IN_ARCHIVE = 'imgs'

SUBSET_TO_LANGUAGE: Dict[str, str] = {
    'en_open': 'en',
    'en_closed': 'en',
    'zh_open': 'zh',
    'zh_closed': 'zh',
}

# The English and Chinese halves ask different questions about the same images, so the answer
# language has to be pinned per sample: targets are stored in the question language.
EN_PROMPT_TEMPLATE = """{question}
Answer the question with a single word or phrase in English.
The last line of your response must be of the form "ANSWER: <answer>" (without quotes)."""

ZH_PROMPT_TEMPLATE = """{question}
请用一个中文词语或短语回答问题。
回答的最后一行必须是 "ANSWER: <答案>" 的形式（不含引号）。"""

DESCRIPTION = """
## Overview

SLAKE is a bilingual (English / Chinese) radiology visual question answering benchmark built by
physicians on CT, MRI and X-Ray images. Questions cover both purely visual properties of the scan
and medical knowledge that has to be recalled on top of what the image shows.

## Task Description

- **Task Type**: Medical visual question answering (free-form short answer)
- **Input**: A radiology image plus a question in English or Chinese
- **Output**: A single word or short phrase, in the language of the question
- **Domain**: Radiology (chest, abdomen, brain, pelvis, neck)

## Key Features

- 2,094 test questions over 180 images, roughly balanced between English (1,061) and Chinese (1,033)
- Every question is labelled `OPEN` (free answer) or `CLOSED` (answer drawn from a small closed set,
  mostly yes/no), which is the breakdown the original paper reports
- Questions span ten semantic types: organ, position, abnormality, knowledge-graph, modality, size,
  plane, quantity, color and shape
- Knowledge-graph questions (`base_type=kvqa`) ask about causes, symptoms, treatments and functions
  that cannot be read off the image

## Evaluation Notes

- Primary metric: **Accuracy** by normalized exact match against the single reference answer
- Reported as four subsets, `<language>_<open|closed>`, grouped into an English and a Chinese
  category; the overall score is the sample-weighted mean
- Normalization lower-cases, removes punctuation and parenthesised asides, maps yes/no synonyms
  onto one label — required because Chinese references express the same polarity as
  是的 / 有 / 包含 / 可以 or 不是 / 没有 / 不包含 / 不可以 — and unifies the X-Ray spellings, including
  the Chinese X光 / X射线, because modality references stay in English in the Chinese half
- Answers are read from the `ANSWER:` line requested by the prompt; when the model does not emit
  one, the whole reply is normalized instead, so a reply that only restates the question scores 0
- Exact match is strict by design, matching the original classification-style evaluation: a
  reference such as `Lung, Spinal Cord`, a knowledge-graph list of treatments, or `T2` answered as
  `T2-weighted` only counts when the model reproduces the reference wording, so open-ended
  accuracy on the knowledge-graph questions is expected to be low
- Images ship as a single `imgs.zip` (about 200 MB) and are read directly from the archive
- [Paper](https://arxiv.org/abs/2102.09542) | [Project page](https://www.med-vqa.com/slake/)
"""


@register_benchmark(
    BenchmarkMeta(
        name='slake',
        pretty_name='SLAKE',
        dataset_id='evalscope/SLAKE',
        tags=[Tags.MULTI_MODAL, Tags.MEDICAL, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2102.09542',
        metric_list=['acc'],
        eval_split='test',
        subset_list=list(SUBSET_TO_LANGUAGE),
    )
)
class SLAKEAdapter(VisionLanguageAdapter):
    """Data adapter for evalscope/SLAKE.

    The dataset ships one JSON file per split plus a single ``imgs.zip``; the split file is read
    directly and split into ``<language>_<answer type>`` subsets, the archive is opened once during
    :meth:`load` and read member by member while samples are built.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = SUBSET_TO_LANGUAGE
        # Archive handle and per-image cache; only valid while load() is running. The 2,094 test
        # questions share 180 images, so encoding each image once keeps one base64 string per image.
        self._image_archive: Optional[zipfile.ZipFile] = None
        self._image_cache: Dict[str, str] = {}

    def load(self) -> Tuple[DatasetDict, None]:
        """Download the split file and image archive, then group records into subsets."""
        unknown = [subset for subset in self.subset_list if subset not in SUBSET_TO_LANGUAGE]
        if unknown:
            raise ValueError(f'Unknown SLAKE subsets {unknown}. Valid subsets are: {list(SUBSET_TO_LANGUAGE)}')

        split_file = f'{self.eval_split}.json'
        dataset_path = resolve_snapshot_or_local_path(self, allow_file_pattern=[split_file, IMAGE_ARCHIVE])
        with open(os.path.join(dataset_path, split_file), 'r', encoding='utf-8') as f:
            records: List[Dict[str, Any]] = json.load(f)

        record_map: Dict[str, List[Dict[str, Any]]] = {subset: [] for subset in self.subset_list}
        for record in records:
            subset = f"{record['q_lang']}_{record['answer_type'].lower()}"
            if subset in record_map:
                record_map[subset].append(record)

        with zipfile.ZipFile(os.path.join(dataset_path, IMAGE_ARCHIVE)) as archive:
            self._image_archive = archive
            dataset_dict = build_dataset_dict_from_record_map(
                record_map=record_map,
                sample_fields=self.record_to_sample,
                location=self.dataset_id,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
                seed=None,
            )
        self._image_archive = None
        self._image_cache = {}

        return dataset_dict, None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one SLAKE question into a multimodal Sample."""
        img_name = record['img_name']
        image_b64 = self._image_cache.get(img_name)
        if image_b64 is None:
            image_bytes = self._image_archive.read(f'{IMAGE_DIR_IN_ARCHIVE}/{img_name}')
            image_b64 = self._image_bytes_to_base64(image_bytes, default_format='jpeg')
            self._image_cache[img_name] = image_b64

        template = ZH_PROMPT_TEMPLATE if record['q_lang'] == 'zh' else EN_PROMPT_TEMPLATE
        content_list: List[Content] = [
            ContentImage(image=image_b64),
            ContentText(text=template.format(question=record['question'].strip())),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['answer'],
            metadata={
                'qid': record['qid'],
                'img_name': img_name,
                'answer_type': record['answer_type'],
                'content_type': record['content_type'],
                'modality': record['modality'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Read the answer from the last ``ANSWER:`` line, or fall back to the whole reply."""
        return parse_answer(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score a prediction by normalized exact match against the reference answer."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value = {'acc': float(normalize_answer(filtered_prediction) == normalize_answer(reference))}
        score.main_score_name = 'acc'
        return score
