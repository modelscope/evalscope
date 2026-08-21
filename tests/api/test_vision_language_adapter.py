import pytest
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ContentAudio, ContentImage, ContentText, ContentVideo
from evalscope.config import TaskConfig
from evalscope.utils.io_utils import bytes_to_base64


class DummyVisionLanguageAdapter(VisionLanguageAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(input='', target='')


@pytest.fixture
def adapter() -> DummyVisionLanguageAdapter:
    return DummyVisionLanguageAdapter(
        benchmark_meta=BenchmarkMeta(name='dummy_vlm', dataset_id='dummy', eval_split='test'),
        task_config=TaskConfig(datasets=['dummy_vlm']),
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
    base64_blob = bytes_to_base64(
        b'not-an-image', format='png', add_header=True, content_type='image', guess_mimetype=True
    )

    assert base64_blob.startswith('data:image/png;base64,')


def test_extract_media_normalizes_hf_image_bytes(adapter: DummyVisionLanguageAdapter, png_bytes: bytes) -> None:
    image_map = adapter._extract_media({'images': [{'bytes': png_bytes}]}, media_type='image')

    assert isinstance(image_map[1], dict) and image_map[1]['url'].startswith('data:image/png;base64,')


def test_extract_media_accepts_api_ready_image_dict(adapter: DummyVisionLanguageAdapter) -> None:
    image_map = adapter._extract_media({'images': [{'url': 'https://example.com/cat.png'}]}, media_type='image')
    content_list = adapter._parse_text_with_media('<image 1> Describe the animal.', image_map=image_map)

    assert any(
        isinstance(content, ContentImage) and content.image == 'https://example.com/cat.png'
        for content in content_list
    )


def test_extract_media_rejects_plural_scalar_container(adapter: DummyVisionLanguageAdapter) -> None:
    with pytest.raises(TypeError):
        adapter._extract_media({'images': 'https://example.com/cat.png'}, media_type='image')


def test_extract_media_rejects_wrong_mime_for_audio_bytes(
    adapter: DummyVisionLanguageAdapter, png_bytes: bytes
) -> None:
    with pytest.raises(ValueError):
        adapter._extract_media({'audios': [{'bytes': png_bytes}]}, media_type='audio')


def test_extract_media_skips_empty_cells(adapter: DummyVisionLanguageAdapter) -> None:
    """Sparse csv/tsv columns yield empty strings, whose placeholders must stay unresolved."""
    record = {'image_1': 'https://example.com/cat.png', 'image_2': '', 'image_3': None}
    image_map = adapter._extract_media(record, media_type='image')
    content_list = adapter._parse_text_with_media('<image 1> vs <image 2> vs <image 3>', image_map=image_map)

    assert set(image_map) == {1}
    assert len([content for content in content_list if isinstance(content, ContentImage)]) == 1


def test_extract_media_raises_on_malformed_empty_dict(adapter: DummyVisionLanguageAdapter) -> None:
    with pytest.raises(ValueError):
        adapter._extract_media({'image_1': {}}, media_type='image')


def test_extract_media_bytes_drops_stale_path(adapter: DummyVisionLanguageAdapter, png_bytes: bytes) -> None:
    image_map = adapter._extract_media({'image_1': {'bytes': png_bytes, 'path': 'stale.png'}}, media_type='image')

    assert image_map[1]['url'].startswith('data:image/png;base64,')
    assert 'path' not in image_map[1]


def test_extract_media_rejects_unsupported_format_hint(adapter: DummyVisionLanguageAdapter) -> None:
    with pytest.raises(ValueError):
        adapter._extract_media({'audio_1': 'crowd.wav', 'audio_1_format': 'flac'}, media_type='audio')


def test_parse_text_warns_once_per_missing_placeholder(adapter: DummyVisionLanguageAdapter) -> None:
    adapter._parse_text_with_media('<image 2> and <image 2> and <image 3>', image_map={})

    assert adapter._missing_media_warned == {'<image 2>', '<image 3>'}


@pytest.mark.parametrize(
    'audio, expected_format',
    [
        ('data:audio/wav;base64,UklGRiQAAABXQVZF', 'wav'),
        ('data:audio/x-wav;base64,UklGRiQAAABXQVZF', 'wav'),
        ('data:audio/mp3;base64,SUQzAwAA', 'mp3'),
        ('data:audio/mpeg;base64,SUQzAwAA', 'mp3'),
        ('https://example.com/crowd.wav', 'wav'),
        ('https://example.com/crowd.wav?token=abc', 'wav'),
    ],
)
def test_content_audio_keeps_declared_format(
    adapter: DummyVisionLanguageAdapter, audio: str, expected_format: str
) -> None:
    audio_map = adapter._extract_media({'audio_1': audio}, media_type='audio')
    content_list = adapter._parse_text_with_media('Hear <audio 1>.', audio_map=audio_map)

    assert [content.format for content in content_list if isinstance(content, ContentAudio)] == [expected_format]


def test_parse_text_with_media_preserves_video_and_audio_metadata(adapter: DummyVisionLanguageAdapter) -> None:
    video_map = adapter._extract_media(
        {'videos': [{'path': 'https://example.com/rally.mov', 'format': 'mov', 'start': 1.25, 'end': 3.5, 'fps': 2}]},
        media_type='video'
    )
    audio_map = adapter._extract_media({'audios': [{'path': 'https://example.com/crowd.wav'}]}, media_type='audio')

    content_list = adapter._parse_text_with_media(
        'Watch <video 1> then hear <audio 1> and answer.',
        video_map=video_map,
        audio_map=audio_map,
    )

    assert isinstance(content_list[0], ContentText)
    assert isinstance(content_list[1], ContentVideo)
    assert content_list[1].video == 'https://example.com/rally.mov'
    assert content_list[1].format == 'mov'
    assert content_list[1].start == 1.25
    assert content_list[1].end == 3.5
    assert content_list[1].fps == 2
    assert any(
        isinstance(content, ContentAudio) and content.audio == 'https://example.com/crowd.wav'
        for content in content_list
    )


# ---------------------------------------------------------------------------
# _resolve_media_placeholders
# ---------------------------------------------------------------------------


class TestResolveMediaPlaceholders:
    """Tests for VisionLanguageAdapter._resolve_media_placeholders()."""

    def test_plain_text_no_media(self, adapter: DummyVisionLanguageAdapter):
        """Messages without any media placeholders pass through unchanged."""
        messages = [{'role': 'user', 'content': 'What is the capital of France?'}]
        result = adapter._resolve_media_placeholders(messages)
        # Content is wrapped in ContentText even without placeholders
        content_list = result[0]['content']
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentText)
        assert content_list[0].text == 'What is the capital of France?'

    def test_single_image_placeholder_with_indexed_column(
        self, adapter: DummyVisionLanguageAdapter, tmp_path: Path
    ):
        """<image 1> placeholder is resolved from image_1 column."""
        image_path = tmp_path / 'dog.jpg'
        PILImage.new(mode='RGB', size=(10, 10), color=(0, 128, 0)).save(image_path)

        messages = [{'role': 'user', 'content': '<image 1> What animal is this?'}]
        image_map = adapter._extract_media({'image_1': str(image_path)}, 'image')
        result = adapter._resolve_media_placeholders(messages, image_map=image_map)

        content_list = result[0]['content']
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentImage)
        assert content_list[0].image == str(image_path)
        assert isinstance(content_list[1], ContentText)
        assert 'What animal is this?' in content_list[1].text

    def test_multiple_image_placeholders(self, adapter: DummyVisionLanguageAdapter, tmp_path: Path):
        """<image 1> and <image 2> placeholders resolved from indexed columns."""
        img1 = tmp_path / 'img1.jpg'
        img2 = tmp_path / 'img2.jpg'
        PILImage.new(mode='RGB', size=(10, 10), color=(255, 0, 0)).save(img1)
        PILImage.new(mode='RGB', size=(10, 10), color=(0, 0, 255)).save(img2)

        messages = [{'role': 'user', 'content': '<image 1> Compare with <image 2> Which is brighter?'}]
        image_map = adapter._extract_media({'image_1': str(img1), 'image_2': str(img2)}, 'image')
        result = adapter._resolve_media_placeholders(messages, image_map=image_map)

        content_list = result[0]['content']
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentImage)
        assert content_list[0].image == str(img1)
        assert isinstance(content_list[1], ContentText)
        assert 'Compare with' in content_list[1].text
        assert isinstance(content_list[2], ContentImage)
        assert content_list[2].image == str(img2)
        assert isinstance(content_list[3], ContentText)
        assert 'Which is brighter?' in content_list[3].text

    def test_image_placeholder_with_images_array(
        self, adapter: DummyVisionLanguageAdapter, png_bytes: bytes
    ):
        """<image 1> placeholder resolved from the plural 'images' column."""
        messages = [{'role': 'user', 'content': '<image 1> What color is the square?'}]
        image_map = adapter._extract_media({'images': [{'bytes': png_bytes}]}, 'image')
        result = adapter._resolve_media_placeholders(messages, image_map=image_map)

        content_list = result[0]['content']
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentImage)
        assert content_list[0].image.startswith('data:image/png;base64,')
        assert isinstance(content_list[1], ContentText)
        assert 'What color is the square?' in content_list[1].text

    def test_placeholder_with_underscore_separator(
        self, adapter: DummyVisionLanguageAdapter, tmp_path: Path
    ):
        """<image_1> (underscore) is also recognized as a placeholder."""
        image_path = tmp_path / 'test.jpg'
        PILImage.new(mode='RGB', size=(10, 10), color=(0, 0, 255)).save(image_path)

        messages = [{'role': 'user', 'content': '<image_1> Describe this image.'}]
        image_map = adapter._extract_media({'image_1': str(image_path)}, 'image')
        result = adapter._resolve_media_placeholders(messages, image_map=image_map)

        content_list = result[0]['content']
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentImage)
        assert content_list[0].image == str(image_path)

    def test_missing_media_placeholder_dropped(
        self, adapter: DummyVisionLanguageAdapter
    ):
        """An unresolved placeholder is dropped with a warning (no crash)."""
        messages = [{'role': 'user', 'content': '<image 1> What is shown here?'}]
        result = adapter._resolve_media_placeholders(messages, image_map={})

        content_list = result[0]['content']
        assert isinstance(content_list, list)
        # Only the text part remains; the placeholder is dropped
        assert len(content_list) == 1
        assert isinstance(content_list[0], ContentText)
        assert 'What is shown here?' in content_list[0].text

    def test_mixed_media_types(self, adapter: DummyVisionLanguageAdapter, tmp_path: Path):
        """<image 1> and <video 1> placeholders resolved from indexed columns."""
        image_path = tmp_path / 'photo.jpg'
        PILImage.new(mode='RGB', size=(10, 10), color=(255, 255, 0)).save(image_path)

        messages = [{'role': 'user', 'content': '<image 1> Watch <video 1> and describe both.'}]
        image_map = adapter._extract_media({'image_1': str(image_path)}, 'image')
        video_map = adapter._extract_media({'video_1': 'https://example.com/clip.mp4'}, 'video')
        result = adapter._resolve_media_placeholders(
            messages, image_map=image_map, video_map=video_map
        )

        content_list = result[0]['content']
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentImage)
        assert content_list[0].image == str(image_path)
        assert isinstance(content_list[1], ContentText)
        assert 'Watch' in content_list[1].text
        assert isinstance(content_list[2], ContentVideo)
        assert content_list[2].video == 'https://example.com/clip.mp4'
        assert isinstance(content_list[3], ContentText)
        assert 'describe both' in content_list[3].text

    def test_system_message_preserved(self, adapter: DummyVisionLanguageAdapter, tmp_path: Path):
        """System messages are not modified; only user messages get placeholder resolution."""
        image_path = tmp_path / 'scene.jpg'
        PILImage.new(mode='RGB', size=(10, 10), color=(0, 255, 0)).save(image_path)

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '<image 1> Describe this scene.'},
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'image_captioning','arguments': '{"image_id": "1"}'},
                    },
                ],
            },
            {'role': 'tool', 'content': 'The image shows a green square.'},
            {'role': 'assistant', 'content': 'The image shows a green square.'},
        ]
        image_map = adapter._extract_media({'image_1': str(image_path)}, 'image')
        result = adapter._resolve_media_placeholders(messages, image_map=image_map)

        assert len(result) == 5
        # System message unchanged
        assert result[0]['content'] == 'You are a helpful assistant.'
        # User message has resolved media
        content_list = result[1]['content']
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentImage)
        assert content_list[0].image == str(image_path)

        # other messages unchanged
        assert result[2]['role'] == 'assistant'
        assert result[2]['tool_calls'][0]['function']['name'] == 'image_captioning'
        assert result[3]['role'] == 'tool'
        assert result[3]['content'] == 'The image shows a green square.'
        assert result[4]['role'] == 'assistant'
        assert result[4]['content'] == 'The image shows a green square.'

    def test_extra_message_keys_preserved(self, adapter: DummyVisionLanguageAdapter, tmp_path: Path):
        """Extra keys on user messages (e.g. name) survive placeholder resolution."""
        image_path = tmp_path / 'keytest.jpg'
        PILImage.new(mode='RGB', size=(10, 10), color=(128, 128, 128)).save(image_path)

        messages = [{'role': 'user', 'name': 'alice', 'content': '<image 1> Describe.'}]
        image_map = adapter._extract_media({'image_1': str(image_path)}, 'image')
        result = adapter._resolve_media_placeholders(messages, image_map=image_map)

        assert result[0]['role'] == 'user'
        assert result[0]['name'] == 'alice'
        assert isinstance(result[0]['content'], list)

    def test_empty_messages_list(self, adapter: DummyVisionLanguageAdapter):
        """An empty messages list returns an empty list."""
        assert adapter._resolve_media_placeholders([]) == []

    def test_non_dict_message_passthrough(self, adapter: DummyVisionLanguageAdapter):
        """Non-dict items in the messages list pass through unchanged."""
        messages = ['not-a-dict', {'role': 'user', 'content': 'hello'}]
        result = adapter._resolve_media_placeholders(messages)
        assert result[0] == 'not-a-dict'
        # Content is wrapped in ContentText even without placeholders
        content_list = result[1]['content']
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentText)
        assert content_list[0].text == 'hello'
