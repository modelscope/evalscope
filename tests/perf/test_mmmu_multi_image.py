import base64
from io import BytesIO

import pytest
from PIL import Image

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.mmmu_multi_image import MMMUMultiImageDatasetPlugin


def _args(dataset_args=None, **kwargs) -> Arguments:
    return Arguments(
        model='test-model',
        url='http://localhost:8080/v1/chat/completions',
        dataset='mmmu_multi_image',
        dataset_args=dataset_args,
        **kwargs,
    )


def _image_bytes(mode: str = 'RGB') -> bytes:
    buffer = BytesIO()
    Image.new(mode, (2, 2), 'white').save(buffer, format='PNG')
    return buffer.getvalue()


class TestMMMUMultiImageDataset:
    def test_builds_one_message_with_multiple_images_in_source_order(self, monkeypatch):
        plugin = MMMUMultiImageDatasetPlugin(_args({'subset': 'Music', 'min_images': 2}))
        rows = [
            {
                'question': 'Compare <image 1> and <image 2>.',
                'options': "['same', 'different']",
                'image_1': Image.new('RGB', (2, 2), 'red'),
                'image_2': {'bytes': _image_bytes()},
                'image_3': None,
            }
        ]
        load_args = {}

        def fake_load_hub_dataset(dataset_id, split='train', subset='default'):
            load_args.update(dataset_id=dataset_id, split=split, subset=subset)
            return rows

        monkeypatch.setattr(plugin, 'load_hub_dataset', fake_load_hub_dataset)

        requests = list(plugin.build_messages())

        assert load_args == {
            'dataset_id': 'AI-ModelScope/MMMU',
            'split': 'validation',
            'subset': 'Music',
        }
        assert len(requests) == 1
        message = requests[0][0]
        assert message['role'] == 'user'
        assert message['content'][0] == {
            'type': 'text',
            'text': "Compare <image 1> and <image 2>.\nOptions: ['same', 'different']",
        }
        assert [part['type'] for part in message['content']] == ['text', 'image_url', 'image_url']
        assert all(part['image_url']['url'].startswith('data:image/') for part in message['content'][1:])

    def test_encodes_alpha_bearing_images_end_to_end(self, monkeypatch):
        # Regression for Yunnglin's blocking review: real AI-ModelScope/MMMU
        # Music validation rows contain RGBA images, and the JPEG default of
        # PIL_to_base64 cannot write alpha-bearing modes. The first MMMU row
        # must produce its request instead of raising mid-iteration.
        plugin = MMMUMultiImageDatasetPlugin(_args({'subset': 'Music', 'min_images': 2}))
        rows = [
            {
                'question': 'Which image matches the score?',
                'options': "['A', 'B']",
                'image_1': Image.new('RGBA', (4, 4), (255, 0, 0, 128)),
                'image_2': Image.new('LA', (4, 4), (120, 200)),
                'image_3': {'bytes': _image_bytes('RGBA')},
                'image_4': Image.new('P', (4, 4)),
                'image_5': Image.new('RGB', (4, 4), 'blue'),
            }
        ]
        monkeypatch.setattr(plugin, 'load_hub_dataset', lambda **_: rows)

        requests = list(plugin.build_messages())

        assert len(requests) == 1
        message = requests[0][0]
        image_parts = [part for part in message['content'] if part['type'] == 'image_url']
        # Image order and count are preserved.
        assert len(image_parts) == 5
        for part in image_parts:
            url = part['image_url']['url']
            # The data-URL MIME must match the encoded bytes.
            assert url.startswith('data:image/jpeg;base64,')
            payload = url.split(',', 1)[1]
            # The declared JPEG MIME is backed by real JPEG bytes.
            assert Image.open(BytesIO(base64.b64decode(payload))).format == 'JPEG'

    def test_skips_rows_below_minimum_image_count(self, monkeypatch):
        plugin = MMMUMultiImageDatasetPlugin(_args({'min_images': 2}))
        monkeypatch.setattr(
            plugin,
            'load_hub_dataset',
            lambda **_: [
                {
                    'question': 'Single image only',
                    'options': '[]',
                    'image_1': Image.new('RGB', (2, 2), 'white'),
                }
            ],
        )

        assert list(plugin.build_messages()) == []

    def test_min_images_is_validated(self):
        with pytest.raises(Exception, match='min_images must be between 2 and 7'):
            MMMUMultiImageDatasetPlugin(_args({'min_images': 1}))

    def test_rejects_tokenized_prompt_mode(self):
        with pytest.raises(ValueError, match='not supported with the mmmu_multi_image dataset'):
            MMMUMultiImageDatasetPlugin(_args(tokenize_prompt=True, tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct'))
