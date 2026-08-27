from typing import Any, List

import pytest
from PIL import Image

from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.model import GenerateConfig
from evalscope.models.image_edit_model import ImageEditAPI
from evalscope.models.text2image_model import Text2ImageAPI
from evalscope.models.utils.openai import openai_chat_completion_part
from evalscope.utils.io_utils import PIL_to_base64


class _FakePipelineResult:

    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


def _install_fake_pipeline(
    monkeypatch: Any,
    attr_name: str,
    image: Image.Image,
    on_call: Any = None,
) -> None:
    class FakePipeline:

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> 'FakePipeline':
            return cls()

        def to(self, device: Any) -> 'FakePipeline':
            return self

        def __call__(self, *args: Any, **kwargs: Any) -> _FakePipelineResult:
            if on_call is not None:
                on_call()
            return _FakePipelineResult(image)

    import modelscope

    monkeypatch.setattr(modelscope, attr_name, FakePipeline, raising=False)


def _generated_image_content(output: Any) -> ContentImage:
    content = output.choices[0].message.content
    assert isinstance(content, list) and isinstance(content[0], ContentImage)
    return content[0]


def test_text2image_output_is_usable_as_openai_chat_input(monkeypatch: Any) -> None:
    image = Image.new('RGB', (8, 8), color='red')
    _install_fake_pipeline(monkeypatch, 'DiffusionPipeline', image)

    api = Text2ImageAPI(model_name='test-diffusion')
    output = api.generate(
        input=[ChatMessageUser(content='a red square')],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(),
    )

    part = openai_chat_completion_part(_generated_image_content(output))
    url = part['image_url']['url']
    assert url.startswith('data:image/')


def test_image_edit_output_is_usable_as_openai_chat_input(monkeypatch: Any) -> None:
    generated = Image.new('RGB', (8, 8), color='blue')
    _install_fake_pipeline(monkeypatch, 'QwenImageEditPipeline', generated)

    source_image = PIL_to_base64(Image.new('RGB', (8, 8), color='green'), format='PNG', add_header=True)
    api = ImageEditAPI(model_name='Qwen-Image-Edit-test')
    output = api.generate(
        input=[
            ChatMessageUser(content=[
                ContentText(text='make it blue'),
                ContentImage(image=source_image),
            ])
        ],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(),
    )

    part = openai_chat_completion_part(_generated_image_content(output))
    url = part['image_url']['url']
    assert url.startswith('data:image/')


def test_text2image_time_is_elapsed_seconds(monkeypatch: Any) -> None:
    clock = [100.0]
    monkeypatch.setattr('evalscope.models.text2image_model.time.monotonic', lambda: clock[0])
    image = Image.new('RGB', (8, 8), color='red')

    def _advance_clock() -> None:
        clock[0] = 100.25

    _install_fake_pipeline(monkeypatch, 'DiffusionPipeline', image, on_call=_advance_clock)

    api = Text2ImageAPI(model_name='test-diffusion')
    output = api.generate(
        input=[ChatMessageUser(content='a red square')],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(),
    )

    # elapsed seconds since generate() started, not a wall-clock epoch timestamp
    assert output.time == pytest.approx(0.25)


def test_image_edit_time_is_elapsed_seconds(monkeypatch: Any) -> None:
    clock = [200.0]
    monkeypatch.setattr('evalscope.models.image_edit_model.time.monotonic', lambda: clock[0])
    generated = Image.new('RGB', (8, 8), color='blue')

    def _advance_clock() -> None:
        clock[0] = 200.5

    _install_fake_pipeline(monkeypatch, 'QwenImageEditPipeline', generated, on_call=_advance_clock)

    source_image = PIL_to_base64(Image.new('RGB', (8, 8), color='green'), format='PNG', add_header=True)
    api = ImageEditAPI(model_name='Qwen-Image-Edit-test')
    output = api.generate(
        input=[
            ChatMessageUser(content=[
                ContentText(text='make it blue'),
                ContentImage(image=source_image),
            ])
        ],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(),
    )

    # elapsed seconds since generate() started, not a wall-clock epoch timestamp
    assert output.time == pytest.approx(0.5)
