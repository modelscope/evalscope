import pytest
from pathlib import Path
from typing import List

from evalscope.api.evaluator import ReviewResult
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.metric import SampleScore, Score
from evalscope.utils.data_utils import _serialize_messages


def _review_result(image: str) -> ReviewResult:
    return ReviewResult(
        index=0,
        target='Dog',
        messages=[ChatMessageUser(content=[ContentText(text='What animal is this?'), ContentImage(image=image)])],
        sample_score=SampleScore(score=Score(value={'acc': 1.0}), sample_id='0'),
    )


def _image_values(review_result: ReviewResult) -> List[str]:
    values = []
    for message in _serialize_messages(review_result):
        content = message['content']
        if isinstance(content, list):
            values.extend(block['image'] for block in content if block['type'] == 'image')
    return values


def test_local_image_path_is_absolutised_for_the_frontend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A relative path cannot be told apart from a base64 payload by the client,
    # so it must be absolutised before it reaches the dashboard.
    image_path = tmp_path / 'dog.jpg'
    image_path.write_bytes(b'fake-jpg')
    monkeypatch.chdir(tmp_path)

    assert _image_values(_review_result('dog.jpg')) == [str(image_path.resolve())]


def test_absolute_image_path_is_preserved(tmp_path: Path) -> None:
    image_path = tmp_path / 'dog.jpg'
    image_path.write_bytes(b'fake-jpg')

    assert _image_values(_review_result(str(image_path))) == [str(image_path)]


def test_data_uri_and_base64_images_are_left_alone() -> None:
    data_uri = 'data:image/png;base64,aGVsbG8='

    assert _image_values(_review_result(data_uri)) == [data_uri]
    assert _image_values(_review_result('aGVsbG8=')) == ['aGVsbG8=']
