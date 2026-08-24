from io import BytesIO
from pathlib import Path
from PIL import Image

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ContentImage, ContentText
from evalscope.api.model import ModelOutput, get_model
from evalscope.benchmarks.mmmu.mmmu_adapter import MULTI_CHOICE_TYPE, MMMUAdapter
from evalscope.config import TaskConfig
from evalscope.models.mockllm import MockLLM


def _adapter() -> MMMUAdapter:
    return MMMUAdapter(
        benchmark_meta=BenchmarkMeta(
            name='mmmu', dataset_id='AI-ModelScope/MMMU', eval_split='validation', metric_list=['acc']
        ),
        task_config=TaskConfig(datasets=['mmmu']),
    )


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format=image_format)
    return buffer.getvalue()


def test_mmmu_preserves_complete_prompt_media_order_and_detects_mime_type() -> None:
    content, answers = _adapter().create_content_and_answers_list({
        'question_type': MULTI_CHOICE_TYPE,
        'question': 'Before <image 1> between <image 2> after.',
        'options': "['A', 'B']",
        'image_1': {'bytes': _image_bytes('PNG')},
        'image_2': {'bytes': _image_bytes('JPEG')},
    })

    assert [item.type for item in content] == ['text', 'image', 'text', 'image', 'text']
    text_segments = [item.text for item in content if isinstance(item, ContentText)]
    assert text_segments[0].endswith('\n\nBefore ')
    assert text_segments[1] == ' between '
    assert text_segments[2] == ' after.\n\nA) A\nB) B'
    images = [item for item in content if isinstance(item, ContentImage)]
    assert [image.image.split(';', 1)[0] for image in images] == ['data:image/png', 'data:image/jpeg']
    assert answers == ['A', 'B']


def test_mmmu_prefers_indexed_media_to_plural_media() -> None:
    content, _ = _adapter().create_content_and_answers_list({
        'question_type': MULTI_CHOICE_TYPE,
        'question': 'Read <image 1>.',
        'options': "['A', 'B']",
        'image_1': {'path': 'indexed.png'},
        'images': [{'path': 'plural.png'}],
    })

    assert [item.image for item in content if isinstance(item, ContentImage)] == ['indexed.png']


def test_mmmu_supports_undecoded_path_image() -> None:
    content, _ = _adapter().create_content_and_answers_list({
        'question_type': MULTI_CHOICE_TYPE,
        'question': 'Read <image 1>.',
        'options': "['A', 'B']",
        'image_1': {'bytes': None, 'path': 'diagram.png'},
    })

    images = [item.image for item in content if isinstance(item, ContentImage)]
    assert images == ['diagram.png']


def test_mmmu_normalized_media_reaches_inference_and_scoring(tmp_path: Path) -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({
        'id': 'sample',
        'question_type': MULTI_CHOICE_TYPE,
        'question': 'Read <image 1>.',
        'options': "['A', 'B']",
        'answer': 'A',
        'subfield': 'Accounting',
        'explanation': '',
        'img_type': 'image',
        'topic_difficulty': 'easy',
        'image_1': {'bytes': _image_bytes('PNG')},
    })
    model = get_model(
        MockLLM(
            model_name='mock-mmmu',
            custom_outputs=[ModelOutput.from_content(model='mock-mmmu', content='ANSWER: A')],
        ),
        eval_type='mock_llm',
        memoize=False,
    )

    task_state = adapter.run_inference(model, sample, str(tmp_path))
    score = adapter.calculate_metrics(task_state).score

    assert task_state.messages[0].content == sample.input[0].content
    assert score.value == {'accuracy': 1.0}
