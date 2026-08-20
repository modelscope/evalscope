# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import pytest
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.benchmarks.general_vqa.general_vqa_adapter import GeneralVQAAdapter
from evalscope.config import TaskConfig


@pytest.fixture
def adapter() -> GeneralVQAAdapter:
    return GeneralVQAAdapter(
        benchmark_meta=BenchmarkMeta(
            name='general_vqa',
            dataset_id='dummy',
            eval_split='test',
        ),
        task_config=TaskConfig(datasets=['general_vqa']),
    )


@pytest.fixture
def png_bytes() -> bytes:
    image = PILImage.new(mode='RGB', size=(10, 10), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


class TestGeneralVQAAdapterRecordToSample:

    def test_tsv_json_string_messages(self, adapter: GeneralVQAAdapter, tmp_path: Path):
        """Messages as a JSON string (TSV format) with placeholders."""
        image_path = tmp_path / 'car.jpg'
        PILImage.new(mode='RGB', size=(10, 10), color=(128, 128, 0)).save(image_path)

        record = {
            'messages': json.dumps([
                {'role': 'user', 'content': '<image 1> What brand is this car?'}
            ]),
            'image_1': str(image_path),
            'answer': 'Tesla',
        }
        sample = adapter.record_to_sample(record)
        content_list = sample.input[0].content
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentImage)
        assert content_list[0].image == str(image_path)
        assert isinstance(content_list[1], ContentText)
        assert 'What brand is this car?' in content_list[1].text

    def test_no_messages_key(self, adapter: GeneralVQAAdapter):
        """Record without 'messages' key returns empty input."""
        record = {'answer': 'Nothing'}
        sample = adapter.record_to_sample(record)
        assert sample.input == []
        assert sample.target == 'Nothing'

    def test_empty_answer(self, adapter: GeneralVQAAdapter):
        """Record with empty answer string."""
        record = {
            'messages': [
                {'role': 'user', 'content': 'Say nothing.'}
            ],
            'answer': '',
        }
        sample = adapter.record_to_sample(record)
        assert sample.target == ''