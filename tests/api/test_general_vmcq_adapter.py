import pytest
from datasets import Dataset, Features, Image, Sequence, Value
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import LocalDataLoader
from evalscope.api.messages import ContentImage, ContentText
from evalscope.benchmarks.general_vmcq.general_vmcq_adapter import GeneralVMCQAdapter
from evalscope.config import TaskConfig
from evalscope.utils.io_utils import bytes_to_base64
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


def test_bytes_to_base64_guess_mimetype_uses_detected_header(jpeg_bytes: bytes) -> None:
    base64_image = bytes_to_base64(jpeg_bytes, add_header=True, guess_mimetype=True)

    assert base64_image.startswith('data:image/jpeg;base64,')


def test_bytes_to_base64_guess_mimetype_falls_back_when_unknown() -> None:
    base64_blob = bytes_to_base64(b'not-an-image', format='png', add_header=True, content_type='image', guess_mimetype=True)

    assert base64_blob.startswith('data:image/png;base64,')


def test_extract_images_converts_hf_bytes_dict(adapter: GeneralVMCQAdapter, png_bytes: bytes) -> None:
    image_map = adapter._extract_images({'images': [{'bytes': png_bytes}]})

    assert image_map[1].startswith('data:image/png;base64,')


def test_extract_images_reads_hf_path_dict(adapter: GeneralVMCQAdapter, tmp_path: Path, jpeg_bytes: bytes) -> None:
    image_path = tmp_path / 'sample.jpg'
    image_path.write_bytes(jpeg_bytes)

    image_map = adapter._extract_images({'image_1': {'path': str(image_path)}})

    assert image_map[1] == str(image_path)


def test_extract_images_rejects_unsupported_value_type(adapter: GeneralVMCQAdapter) -> None:
    with pytest.raises(TypeError):
        adapter._extract_images({'image_1': 123})


def test_extract_images_rejects_dict_without_path_or_bytes(adapter: GeneralVMCQAdapter) -> None:
    with pytest.raises(ValueError):
        adapter._extract_images({'image_1': {'url': 'https://example.com/image.png'}})


def test_create_content_and_answers_list_supports_images_array(
    adapter: GeneralVMCQAdapter, png_bytes: bytes
) -> None:
    content_list, answers_list = adapter.create_content_and_answers_list(
        {
            'question': '<image 1> What color is the square?',
            'options': ['Red', 'Blue', 'Green', 'Yellow'],
            'images': [{'bytes': png_bytes}],
            'answer': 'A',
        }
    )

    image_content = [content for content in content_list if isinstance(content, ContentImage)]
    text_content = [content for content in content_list if isinstance(content, ContentText)]

    assert answers_list == ['Red', 'Blue', 'Green', 'Yellow']
    assert len(image_content) == 1
    assert image_content[0].image.startswith('data:image/png;base64,')
    assert any('What color is the square?' in content.text for content in text_content)


def test_local_loader_supports_parquet_with_hf_image_column(
    adapter: GeneralVMCQAdapter, tmp_path: Path, png_bytes: bytes
) -> None:
    image_path = tmp_path / 'sample.png'
    parquet_path = tmp_path / 'general_vmcq.parquet'
    image_path.write_bytes(png_bytes)

    dataset = Dataset.from_dict(
        {
            'question': ['<image 1> What color is the square?'],
            'options': [['Red', 'Blue', 'Green', 'Yellow']],
            'image_1': [str(image_path)],
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
        isinstance(content, ContentImage) and content.image == str(image_path)
        for content in content_list
    )
