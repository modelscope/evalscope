import pytest
from io import BytesIO
from PIL import Image as PILImage

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ContentAudio, ContentImage, ContentText, ContentVideo
from evalscope.config import TaskConfig
from evalscope.utils.io_utils import bytes_to_base64


class DummyVisionLanguageAdapter(VisionLanguageAdapter):

	def record_to_sample(self, record):
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
	base64_blob = bytes_to_base64(b'not-an-image', format='png', add_header=True, content_type='image', guess_mimetype=True)

	assert base64_blob.startswith('data:image/png;base64,')


def test_extract_media_normalizes_hf_image_bytes(adapter: DummyVisionLanguageAdapter, png_bytes: bytes) -> None:
    image_map = adapter._extract_media(
        {'images': [{'bytes': png_bytes}]}, mtype='image'
    )

    assert isinstance(image_map[1], dict) and image_map[1]['url'].startswith(
        'data:image/png;base64,'
    )

def test_extract_media_accepts_api_ready_image_dict(adapter: DummyVisionLanguageAdapter) -> None:
	image_map = adapter._extract_media({'images': [{'url': 'https://example.com/cat.png'}]}, mtype='image')
	content_list = adapter._parse_text_with_media('<image 1> Describe the animal.', image_map=image_map)

	assert any(isinstance(content, ContentImage) and content.image == 'https://example.com/cat.png' for content in content_list)


def test_extract_media_rejects_plural_scalar_container(adapter: DummyVisionLanguageAdapter) -> None:
	with pytest.raises(TypeError):
		adapter._extract_media({'images': 'https://example.com/cat.png'}, mtype='image')


def test_extract_media_rejects_wrong_mime_for_audio_bytes(
	adapter: DummyVisionLanguageAdapter, png_bytes: bytes
) -> None:
	with pytest.raises(ValueError):
		adapter._extract_media({'audios': [{'bytes': png_bytes}]}, mtype='audio')


def test_parse_text_with_media_preserves_video_and_audio_metadata(adapter: DummyVisionLanguageAdapter) -> None:
	video_map = adapter._extract_media(
		{
			'videos': [
				{'path': 'https://example.com/rally.mov', 'format': 'mov', 'start': 1.25, 'end': 3.5, 'fps': 2}
			]
		},
		mtype='video',
	)
	audio_map = adapter._extract_media({'audios': [{'path': 'https://example.com/crowd.wav'}]}, mtype='audio')

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
	assert any(isinstance(content, ContentAudio) and content.audio == 'https://example.com/crowd.wav' for content in content_list)
