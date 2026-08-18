import json
import pytest
from datasets import Dataset, Features, Image, Sequence, Value
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import LocalDataLoader
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentImage, ContentText
from evalscope.benchmarks.general_vmcq.general_vmcq_adapter import GeneralVMCQAdapter
from evalscope.config import TaskConfig
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@pytest.fixture
def adapter() -> GeneralVMCQAdapter:
    return GeneralVMCQAdapter(
        benchmark_meta=BenchmarkMeta(
            name='general_vmcq',
            dataset_id='dummy',
            eval_split='test',
            prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
        ),
        task_config=TaskConfig(datasets=['general_vmcq']),
    )


@pytest.fixture
def png_bytes() -> bytes:
    image = PILImage.new(mode='RGB', size=(10, 10), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    image = PILImage.new(mode='RGB', size=(10, 10), color=(0, 0, 255))
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()


def test_record_to_sample_builds_realistic_image_question(adapter: GeneralVMCQAdapter, tmp_path: Path) -> None:
    image_path = tmp_path / 'traffic_stop.jpg'
    sample = adapter.record_to_sample(
        {
            'id': 'traffic-sign-1',
            'question': '<image 1> Which traffic sign is shown?',
            'options': ['Stop', 'Yield', 'No parking', 'School zone'],
            'image_1': str(image_path),
            'answer': 'A',
        }
    )

    assert isinstance(sample.input[0], ChatMessageUser)
    content_list = sample.input[0].content
    joined_text = ''.join(content.text for content in content_list if isinstance(content, ContentText))

    assert sample.target == 'A'
    assert sample.choices == ['Stop', 'Yield', 'No parking', 'School zone']
    assert any(isinstance(content, ContentImage) and content.image == str(image_path) for content in content_list)
    assert 'Which traffic sign is shown?' in joined_text
    assert 'Stop' in joined_text


def test_record_to_sample_supports_json_options_and_audio_placeholder(adapter: GeneralVMCQAdapter) -> None:
    sample = adapter.record_to_sample(
        {
            'question': '<audio 1> Which sport is being played?',
            'options': json.dumps(['Tennis', 'Basketball', 'Baseball', 'Swimming']),
            'audio_1': 'https://example.com/crowd.wav',
            'answer': 'B',
        }
    )

    assert isinstance(sample.input[0], ChatMessageUser)
    content_list = sample.input[0].content
    text_segments = ''.join(content.text for content in content_list if isinstance(content, ContentText))

    assert sample.target == 'B'
    assert any(isinstance(content, ContentAudio) and content.format == 'wav' for content in content_list)
    assert 'Basketball' in text_segments


def test_create_content_and_answers_list_supports_images_array(
    adapter: GeneralVMCQAdapter, png_bytes: bytes
) -> None:
    sample = adapter.record_to_sample(
        {
            'question': '<image 1> What color is the square?',
            'options': ['Red', 'Blue', 'Green', 'Yellow'],
            'images': [{'bytes': png_bytes}],
            'answer': 'A',
        }
    )

    assert isinstance(sample.input[0], ChatMessageUser)
    content_list = sample.input[0].content
    image_content = [content for content in content_list if isinstance(content, ContentImage)]
    text_content = [content for content in content_list if isinstance(content, ContentText)]

    assert sample.choices == ['Red', 'Blue', 'Green', 'Yellow']
    assert len(image_content) == 1
    assert image_content[0].image.startswith('data:image/png;base64,')
    assert any('What color is the square?' in content.text for content in text_content)


def test_local_loader_supports_parquet_with_hf_image_column(
    adapter: GeneralVMCQAdapter, tmp_path: Path, png_bytes: bytes
) -> None:
    parquet_path = tmp_path / 'general_vmcq.parquet'
    dataset = Dataset.from_dict(
        {
            'question': ['<image 1> What color is the square?'],
            'options': [['Red', 'Blue', 'Green', 'Yellow']],
            'image_1': [{'bytes': png_bytes}],
            'answer': ['A'],
        },
        features=Features(
            {
                'question': Value('string'),
                'options': Sequence(Value('string')),
                'image_1': Image(),
                'answer': Value('string'),
            }
        ),
    )
    dataset.to_parquet(str(parquet_path))

    loaded_dataset = LocalDataLoader(
        data_id_or_path=str(parquet_path),
        split='test',
        subset='default',
        sample_fields=adapter.record_to_sample,
    ).load()
    content_list = loaded_dataset[0].input[0].content

    assert any(isinstance(content, ContentImage) for content in content_list)
    assert any(
        isinstance(content, ContentImage) and content.image.startswith('data:image/png;base64,')
        for content in content_list
    )
