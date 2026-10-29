import base64
from pathlib import Path

import pytest

from evalscope.api.messages import ChatMessageUser, ContentImage


@pytest.mark.parametrize('scheme', ['http', 'https'])
def test_image_urls_are_sent_as_url_sources(scheme: str) -> None:
    pytest.importorskip('anthropic')
    from evalscope.models.utils.anthropic import anthropic_chat_messages

    url = f'{scheme}://example.com/image.png?version=1'
    _, messages = anthropic_chat_messages(
        [ChatMessageUser(content=[ContentImage(image=url, internal={'anthropic': {'cache_control': {'type': 'ephemeral'}}})])]
    )

    assert messages[0]['content'] == [
        {'type': 'image', 'source': {'type': 'url', 'url': url}, 'cache_control': {'type': 'ephemeral'}}
    ]


@pytest.mark.parametrize('source_kind', ['path', 'data_uri'])
def test_local_images_remain_base64_sources(source_kind: str, tmp_path: Path) -> None:
    pytest.importorskip('anthropic')
    from evalscope.models.utils.anthropic import anthropic_chat_messages

    image_data = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='
    )
    encoded = base64.b64encode(image_data).decode('ascii')
    if source_kind == 'path':
        image_path = tmp_path / 'image.png'
        image_path.write_bytes(image_data)
        source = str(image_path)
    else:
        source = f'data:image/png;base64,{encoded}'

    _, messages = anthropic_chat_messages([ChatMessageUser(content=[ContentImage(image=source)])])

    assert messages[0]['content'] == [
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': encoded}}
    ]
