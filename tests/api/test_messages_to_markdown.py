from pathlib import Path

import pytest

from evalscope.api.messages import (
    ChatMessageUser,
    ContentAudio,
    ContentImage,
    ContentText,
    ContentVideo,
    messages_to_markdown,
)


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


def test_relative_audio_path_is_absolutised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The structured chain absolutises image/audio/video alike; the markdown chain must match,
    # otherwise a relative path is indistinguishable from a base64 payload downstream.
    audio_path = tmp_path / 'clip.wav'
    audio_path.write_bytes(b'fake-wav')
    monkeypatch.chdir(tmp_path)
    messages = [ChatMessageUser(content=[ContentAudio(audio='clip.wav', format='wav')])]

    markdown = messages_to_markdown(messages)

    assert f"<audio controls src='{audio_path.resolve()}'></audio>" in markdown


def test_relative_video_path_is_absolutised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path = tmp_path / 'clip.mp4'
    video_path.write_bytes(b'fake-mp4')
    monkeypatch.chdir(tmp_path)
    messages = [ChatMessageUser(content=[ContentVideo(video='clip.mp4', format='mp4')])]

    markdown = messages_to_markdown(messages)

    assert f"<video controls src='{video_path.resolve()}'></video>" in markdown


def test_local_audio_path_is_not_truncated_by_max_length(tmp_path: Path) -> None:
    # `Sample.pretty_print` defaults to max_length=50, which is shorter than most paths.
    audio_path = tmp_path / 'a-long-name-clip.wav'
    audio_path.write_bytes(b'fake-wav')
    messages = [ChatMessageUser(content=[ContentAudio(audio=str(audio_path), format='wav')])]

    markdown = messages_to_markdown(messages, max_length=10)

    assert f"<audio controls src='{audio_path}'></audio>" in markdown


def test_local_video_path_is_not_truncated_by_max_length(tmp_path: Path) -> None:
    video_path = tmp_path / 'a-long-name-clip.mp4'
    video_path.write_bytes(b'fake-mp4')
    messages = [ChatMessageUser(content=[ContentVideo(video=str(video_path), format='mp4')])]

    markdown = messages_to_markdown(messages, max_length=10)

    assert f"<video controls src='{video_path}'></video>" in markdown


def test_base64_audio_and_video_are_still_truncated_by_max_length() -> None:
    audio_messages = [ChatMessageUser(content=[ContentAudio(audio='a' * 100, format='wav')])]
    video_messages = [ChatMessageUser(content=[ContentVideo(video='b' * 100, format='mp4')])]

    assert f"<audio controls src='{'a' * 10}'></audio>" in messages_to_markdown(audio_messages, max_length=10)
    assert f"<video controls src='{'b' * 10}'></video>" in messages_to_markdown(video_messages, max_length=10)
