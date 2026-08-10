# flake8: noqa: E501
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, Sample, load_local_file_dataset, resolve_snapshot_or_local_path
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt

CHOICE_LETTERS = ['A', 'B', 'C', 'D']
DATA_FILE = 'test_clean.csv'
IMAGE_ARCHIVE = 'images.zip'
IMAGE_DIR_IN_ARCHIVE = 'images'

DESCRIPTION = """
## Overview

PMC-VQA is a large-scale medical visual question answering benchmark built from figures of
biomedical papers in the PubMed Central Open Access subset. This integration evaluates the
manually verified **test_clean** split, the 2,000-question subset the authors recommend for
reporting results.

## Task Description

- **Task Type**: Medical Visual Question Answering (single-answer multiple choice)
- **Input**: A biomedical figure plus a question with four candidate answers
- **Output**: A single answer letter (A/B/C/D)
- **Domain**: Medicine and biomedical imaging (radiology, pathology, microscopy, plus charts and diagrams found in papers)

## Key Features

- 2,000 questions over 1,440 distinct figures, each with exactly four answer options
- Questions were generated from figure captions and then manually verified, so test_clean is
  substantially cleaner than the raw 50k test split
- Covers a wide range of imaging modalities and diseases, as well as non-photographic figures
  such as plots and diagrams
- Requires reading fine-grained visual detail together with biomedical domain knowledge

## Evaluation Notes

- Primary metric: **Accuracy** over the four options
- Answers are extracted from the `ANSWER: [LETTER]` line requested by the prompt; the original
  paper instead matches free-form generations to the closest option string, which is only needed
  for models that cannot follow an answer format
- Images are shipped as a single `images.zip` (about 18 GB) in the dataset repository. It is
  downloaded once and the figures needed for the evaluated samples are read directly from the
  archive, so no extracted copy is kept on disk
- [Paper](https://arxiv.org/abs/2305.10415) | [GitHub](https://github.com/xiaoman-zhang/PMC-VQA)
"""


def strip_choice_prefix(choice: str, letter: str) -> str:
    """Remove the leading ``<letter>:`` marker from a raw choice string."""
    return re.sub(rf'^{letter}\s*:', '', choice.strip()).strip()


@register_benchmark(
    BenchmarkMeta(
        name='pmc_vqa',
        pretty_name='PMC-VQA',
        dataset_id='evalscope/PMC-VQA',
        tags=[Tags.MULTI_MODAL, Tags.MEDICAL, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2305.10415',
        metric_list=['acc'],
        eval_split='test_clean',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class PMCVQAAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    """Data adapter for evalscope/PMC-VQA (test_clean split).

    The dataset repository ships several CSV files with conflicting schemas, so the standard
    remote loader cannot be used; ``test_clean.csv`` is loaded directly instead. Figures live in
    a single large ``images.zip``, which is opened once during :meth:`load` and read member by
    member while samples are built.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Archive handle used by record_to_sample; only valid while load() is running.
        self._image_archive: Optional[zipfile.ZipFile] = None

    def load(self) -> Tuple[DatasetDict, None]:
        """Download the metadata CSV and image archive, then build the test dataset."""
        dataset_path = resolve_snapshot_or_local_path(self, allow_file_pattern=[DATA_FILE, IMAGE_ARCHIVE])

        with zipfile.ZipFile(os.path.join(dataset_path, IMAGE_ARCHIVE)) as archive:
            self._image_archive = archive
            dataset = load_local_file_dataset(
                adapter=self,
                dataset_path=os.path.join(dataset_path, DATA_FILE),
                subset=self.default_subset,
                split=self.eval_split,
                sample_fields=self.record_to_sample,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
            )
        self._image_archive = None

        return DatasetDict({self.default_subset: dataset}), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a ``test_clean.csv`` row into a multimodal multiple-choice Sample."""
        figure_path = record['Figure_path'].strip()
        image_bytes = self._image_archive.read(f'{IMAGE_DIR_IN_ARCHIVE}/{figure_path}')
        image_b64 = self._image_bytes_to_base64(image_bytes, default_format='jpeg')

        # Options carry a redundant letter prefix, e.g. ' B:Magnetic resonance imaging ',
        # which would be duplicated by the multiple-choice prompt template.
        choices = [strip_choice_prefix(record[f'Choice {letter}'], letter) for letter in CHOICE_LETTERS]
        input_text = prompt(
            question=record['Question'].strip(),
            choices=choices,
            template=self.prompt_template,
        )

        content_list: List[Content] = [
            ContentImage(image=image_b64),
            ContentText(text=input_text),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=choices,
            target=record['Answer_label'].strip(),
            metadata={'figure_path': figure_path},
        )
