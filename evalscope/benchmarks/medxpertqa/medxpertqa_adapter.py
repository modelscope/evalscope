# flake8: noqa: E501
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, Sample, download_dataset_file
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt

SUBSET_LIST = ['Text', 'MM']
IMAGE_ARCHIVE = 'images.zip'
IMAGE_DIR_IN_ARCHIVE = 'images'
ANSWER_CHOICES_MARKER = '\nAnswer Choices:'

DESCRIPTION = """
## Overview

MedXpertQA is an expert-level medical multiple-choice benchmark designed to evaluate advanced
medical knowledge and reasoning. It contains separate text-only and multimodal tracks built from
challenging medical examination questions and reviewed by licensed physicians.

## Task Description

- **Task Type**: Single-answer medical multiple choice
- **Input**: A clinical or biomedical question with answer choices, optionally accompanied by up to six images
- **Output**: One answer letter (A-J for Text or A-E for MM)
- **Domain**: Medicine across 17 specialties and 11 human body systems

## Key Features

- The test split contains 4,450 questions: 2,450 Text questions with ten options and 2,000 MM questions with five options
- The MM track contains radiology, pathology, optical, photographic, diagram, chart, table, document, and vital-sign imagery
- Questions are annotated by medical task, body system, and question type; 3,307 test questions require reasoning and 1,143 assess understanding
- Questions underwent difficulty filtering, option augmentation, leakage mitigation, and multiple rounds of expert review

## Evaluation Notes

- Primary metric: **Accuracy** by exact match of the predicted option letter
- The default prompt uses EvalScope's zero-shot chain-of-thought template, preserving the official step-by-step instruction and exact answer-letter scoring
- Set `max_tokens` high enough for the model to emit the required final `ANSWER: [LETTER]` line; truncated reasoning may otherwise fall back to the shared parser's last valid uppercase letter
- Results are reported separately for the Text and MM subsets and combined with sample-weighted aggregation
- MM images are stored in `images.zip` (about 517 MB) and read directly from the archive without extracting a second copy
- The published dataset has 4,460 records including ten development examples; this integration evaluates the 4,450 held-out test questions
- [Paper](https://arxiv.org/abs/2501.18362) | [GitHub](https://github.com/TsinghuaC3I/MedXpertQA)
"""


@register_benchmark(
    BenchmarkMeta(
        name='medxpertqa',
        pretty_name='MedXpertQA',
        dataset_id='evalscope/MedXpertQA',
        subset_list=SUBSET_LIST,
        tags=[Tags.MULTI_MODAL, Tags.MEDICAL, Tags.MULTIPLE_CHOICE, Tags.REASONING],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2501.18362',
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
        system_prompt='You are a helpful medical assistant.',
        evaluation_version='v1.0',
    )
)
class MedXpertQAAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    """Data adapter for the Text and MM tracks of MedXpertQA."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._image_archive: Optional[zipfile.ZipFile] = None

    def load(self) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        """Open the image archive when the MM subset is requested, then use the standard dataset flow."""
        if 'MM' not in self.subset_list:
            return super().load()

        archive_path = download_dataset_file(
            data_id_or_path=self.dataset_id,
            file_path=IMAGE_ARCHIVE,
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        )
        with zipfile.ZipFile(archive_path) as archive:
            self._image_archive = archive
            try:
                return super().load()
            finally:
                self._image_archive = None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one text-only or multimodal question into a multiple-choice sample."""
        question = record['question'].partition(ANSWER_CHOICES_MARKER)[0].strip()
        choices = [record['options'][letter].strip() for letter in sorted(record['options'])]
        input_text = prompt(question=question, choices=choices, template=self.prompt_template)
        content_list: List[Content] = [ContentText(text=input_text)]

        image_names = record.get('images', [])
        if image_names:
            if self._image_archive is None:
                raise RuntimeError('MedXpertQA image archive is not open while loading an MM sample.')
            for image_name in image_names:
                image_bytes = self._image_archive.read(f'{IMAGE_DIR_IN_ARCHIVE}/{image_name}')
                content_list.append(ContentImage(image=self._image_bytes_to_base64(image_bytes, guess_mimetype=True)))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=choices,
            target=record['label'].strip(),
            metadata={
                'id': record['id'],
                'medical_task': record['medical_task'],
                'body_system': record['body_system'],
                'question_type': record['question_type'],
                'images': image_names,
            },
        )
