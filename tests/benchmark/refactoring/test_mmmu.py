import ast
import pytest
from datasets import Dataset, Image
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset.hub import DatasetHub, HubType
from evalscope.api.messages.content import ContentImage, ContentText
from evalscope.benchmarks.mmmu.mmmu_adapter import (
    MULT_CHOICE_PROMPT,
    MULTI_CHOICE_TYPE,
    OPEN_PROMPT,
    SUBSET_LIST,
    MMMUAdapter,
)
from evalscope.config import TaskConfig
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.multi_choices import prompt

MMMU_MS = 'AI-ModelScope/MMMU'
MMMU_HF = 'MMMU/MMMU'


def _old_create_content_and_answers(
    record: Dict[str, Any],
    adapter: MMMUAdapter,
) -> tuple:
    """Replicate the old ``create_content_and_answers_list`` exactly."""
    question_type = record['question_type']
    image_map: Dict[int, str] = {}
    for i in range(MMMUAdapter.MAX_IMAGES):
        image = record.get(f'image_{i + 1}')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            image_map[i + 1] = image_base64

    if question_type == MULTI_CHOICE_TYPE:
        answers_list: List[str] = ast.literal_eval(record['options'])
        full_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT)
        content_list = adapter._parse_text_with_images(full_text, image_map)
    else:
        answers_list: List[str] = []
        full_text = OPEN_PROMPT.format(question=record['question'])
        content_list = adapter._parse_text_with_images(full_text, image_map)

    return content_list, answers_list


def _load_mmmu_records(subject: str, limit: int) -> List[Dict[str, Any]]:
    dataset: Dataset = DatasetHub(
        data_id_or_path=MMMU_MS,
        data_source=HubType.MODELSCOPE,
    ).load(split='validation', subset=subject)
    for colname, coltype in dataset.features.items():
        if isinstance(coltype, Image):
            dataset = dataset.cast_column(colname, Image(decode=False))
    return [dict(row) for row in dataset.select(range(min(limit, len(dataset))))]


@pytest.fixture()
def mmmu_adapter() -> MMMUAdapter:
    return MMMUAdapter(
        benchmark_meta=BenchmarkMeta(
            name='mmmu',
            pretty_name='MMMU',
            dataset_id=MMMU_MS,
        ),
        task_config=TaskConfig(datasets=['mmmu']),
    )


# we hope to demonstrate that the new function behaves exactly like the old one
@pytest.mark.parametrize('subject', SUBSET_LIST)
def test_content_and_answers_match(
    subject: str,
    mmmu_adapter: MMMUAdapter,
    limit: int = 10,
) -> None:
    """Verify MMMU adapter maintains the exactly same data input to VLM
    before/after ``_extract_media()`` refactoring.
    """
    try:
        records = _load_mmmu_records(subject, limit=limit)
    except Exception as e:
        pytest.skip(f'Cannot load MMMU/{subject}: {e}')

    for record in records:
        old_content, old_answers = _old_create_content_and_answers(record, mmmu_adapter)
        new_content, new_answers = mmmu_adapter.create_content_and_answers_list(record)

        assert len(old_content) == len(new_content)
        assert old_answers == new_answers

        for old_c, new_c in zip(old_content, new_content):
            assert isinstance(new_c, type(old_c))
            if isinstance(old_c, ContentText):
                assert old_c.text == new_c.text
            elif isinstance(old_c, ContentImage):
                assert old_c.image == new_c.image
