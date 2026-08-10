import pytest
from pathlib import Path

from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText, messages_to_markdown


def test_local_image_path_is_emitted_as_an_absolute_path(tmp_path: Path) -> None:
    # Regression test: the legacy `gradio_api/file=` prefix is no longer
    # understood by any renderer, so a local file must be emitted as a path.
    image_path = tmp_path / 'screenshot.png'
    image_path.write_bytes(b'fake-png')
    messages = [ChatMessageUser(content=[ContentText(text='Look:'), ContentImage(image=str(image_path))])]

    markdown = messages_to_markdown(messages)

    assert f'![image](<{image_path}>)' in markdown
    assert 'gradio_api' not in markdown


def test_local_image_path_with_spaces_stays_a_single_destination(tmp_path: Path) -> None:
    # An unwrapped destination containing a space does not parse as a markdown
    # image at all, so the <> form is required here.
    image_dir = tmp_path / 'my images'
    image_dir.mkdir()
    image_path = image_dir / 'a shot.png'
    image_path.write_bytes(b'fake-png')
    messages = [ChatMessageUser(content=[ContentImage(image=str(image_path))])]

    markdown = messages_to_markdown(messages)

    assert f'![image](<{image_path}>)' in markdown


def test_relative_image_path_is_absolutised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / 'frame.jpg'
    image_path.write_bytes(b'fake-jpg')
    monkeypatch.chdir(tmp_path)
    messages = [ChatMessageUser(content=[ContentImage(image='frame.jpg')])]

    markdown = messages_to_markdown(messages)

    assert f'![image](<{image_path.resolve()}>)' in markdown


def test_data_uri_image_is_passed_through() -> None:
    data_uri = 'data:image/png;base64,aGVsbG8='
    messages = [ChatMessageUser(content=[ContentImage(image=data_uri)])]

    markdown = messages_to_markdown(messages)

    assert f'![image]({data_uri})' in markdown


def test_base64_image_is_truncated_by_max_length() -> None:
    messages = [ChatMessageUser(content=[ContentImage(image='a' * 100)])]

    markdown = messages_to_markdown(messages, max_length=10)

    assert f'![image]({"a" * 10})' in markdown
