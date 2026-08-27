import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from datasets import Audio, Dataset, DatasetInfo, Features, Image, Sequence, Video, concatenate_datasets
from datasets.table import InMemoryTable
from PIL import Image as PILImage

from evalscope.utils.io_utils import jsonl_to_list, undecode_media


def _generate_jpeg_bytes(dim: int, quality: int = 80) -> bytes:
    """Return a square random-noise JPEG as raw bytes."""
    img = PILImage.fromarray(np.random.randint(0, 256, (dim, dim, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def gen_dataset(
    media_factory: dict,
    extra_dict: dict | None = None,
    num_rows: int = 1,
) -> Dataset:
    features = {}
    row = {}

    # make media cols
    for col_name, (media_bytes, inner_feat, count) in media_factory.items():
        if count == 1:
            features[col_name] = inner_feat
            row[col_name] = {'bytes': media_bytes, 'path': None}
        else:
            features[col_name] = Sequence(inner_feat)
            row[col_name] = [{'bytes': media_bytes, 'path': None}] * count

    # inject non-media cols
    if extra_dict:
        features |= Dataset.from_list([extra_dict]).features
        row.update(extra_dict)

    dataset_features = Features(features)
    table = pa.Table.from_pylist([row], schema=dataset_features.arrow_schema)
    single_row = Dataset(InMemoryTable(table), info=DatasetInfo(features=dataset_features))
    return concatenate_datasets([single_row] * num_rows)


class TestArrowOffsetOverflow:
    """The default ``batch_size=1000`` in ``datasets`` can overflow Arrow's
    32-bit offset when a dataset contains many large images per row.

    These tests reproduce the overflow and confirm that a smaller batch
    size avoids it.
    """

    IMAGES_PER_ROW = 100

    @pytest.mark.parametrize(
        'dim, batch_size',
        [(512, None), (1024, 10)],
    )
    def test_default_batch_overflows_but_small_batch_succeeds(
        self,
        dim: int,
        batch_size: int | None,
    ):
        ds = gen_dataset(
            media_factory={'images': (_generate_jpeg_bytes(dim), Image(decode=True), self.IMAGES_PER_ROW)},
            num_rows=1000,
        )

        # expected overflow with default batch size (1k)
        with pytest.raises(ValueError, match='offset'):
            undecode_media(ds, media_type=['image'], batch_size=1000)

        # A smaller batch size avoids the overflow. One full batch is sufficient to verify the boundary.
        safe_batch_size = batch_size or 100
        result = undecode_media(
            ds.select(range(safe_batch_size)), media_type=['image'], batch_size=batch_size
        )
        assert result.features['images'].feature.decode is False


class TestUndecodeMediaIntegration:
    """End-to-end tests for ``undecode_media``.

    Every test verifies that:
    • target media columns get ``decode=False``
    • non-media columns keep their original feature type and data
    """

    @pytest.mark.parametrize(
        'example',
        [
            {'text': 'hello', 'label': 0},
            {'messages': [{'role': 'user', 'content': 'world'}], 'answer': [1]},
            {'tokens': [1, 2, 3, 4], 'answer': {'text': 'hello'}},
        ],
    )
    def test_leave_non_media_columns_unchanged(self, example: dict):
        """Given a dataset with only text / numeric columns,
        when undecode_media is called,
        then the original object is returned and features are preserved."""
        ds = Dataset.from_list([example])
        undecoded_ds = undecode_media(ds, media_type=['image', 'audio', 'video'])

        assert undecoded_ds.features == ds.features
        assert undecoded_ds[0] == example

    @pytest.mark.parametrize(
        'media_factory, non_media_data',
        [
            pytest.param(
                {'image': (_generate_jpeg_bytes(40), Image(decode=True), 1)},
                {'text': 'cat'},
                id='plain_Image',
            ),
            pytest.param(
                {'images': (_generate_jpeg_bytes(20), Image(decode=True), 4)},
                {'text': 'album'},
                id='Sequence_Image',
            ),
            pytest.param(
                {'audio': (b'dummy', Audio(decode=True), 1)},
                {'text': 'recording'},
                id='plain_Audio',
            ),
            pytest.param(
                {'audios': (b'dummy', Audio(decode=True), 2)},
                {'text': 'playlist'},
                id='Sequence_Audio',
            ),
            pytest.param(
                {'video': (b'dummy', Video(decode=True), 1)},
                {'text': 'clip'},
                id='plain_Video',
            ),
            pytest.param(
                {'videos': (b'dummy', Video(decode=True), 2)},
                {'text': 'clips'},
                id='Sequence_Video',
            ),
            pytest.param(
                {
                    'image': (_generate_jpeg_bytes(40), Image(decode=True), 1),
                    'audio': (b'dummy', Audio(decode=True), 1),
                },
                {'text': 'multi'},
                id='multiple_media_types',
            ),
        ],
    )
    def test_it_disables_decode_on_media_columns_and_preserves_others(
        self,
        media_factory: dict,
        non_media_data: dict,
    ):
        """Given a dataset with a media column (decode=True) and a text column,
        when undecode_media is called with the relevant media_type(s),
        then the media column's decode flag is set to False,
        and the text column is untouched."""
        ds = gen_dataset(media_factory, extra_dict=non_media_data)
        result = undecode_media(ds, media_type=['image', 'audio', 'video'])

        # -- Media columns: decode must be False --
        for col_name in media_factory:
            media_feat = result.features[col_name]
            if isinstance(media_feat, Sequence):
                assert isinstance(media_feat.feature, (Image, Audio, Video))
                assert media_feat.feature.decode is False
            else:
                assert isinstance(media_feat, (Image, Audio, Video))
                assert media_feat.decode is False

        for other_col in non_media_data:
            assert other_col in result.features, f'Column {other_col} missing'
            assert result.features[other_col] == ds.features[other_col]

        original_row = ds.to_list()[0]
        result_row = result.to_list()[0]
        for other_col in non_media_data:
            assert result_row[other_col] == original_row[other_col]


class TestJsonlToList:
    @staticmethod
    def _write(path: Path, content: str) -> str:
        path.write_text(content, encoding='utf-8')
        return str(path)

    def test_tolerant_read_skips_torn_tail(self, tmp_path: Path) -> None:
        file_path = self._write(tmp_path / 'torn.jsonl', '{"a": 1}\n{"a": 2}\n{"a": 3')

        records = jsonl_to_list(file_path, skip_invalid=True)

        assert records == [{'a': 1}, {'a': 2}]

    def test_tolerant_read_skips_null_empty_and_non_dict_json(self, tmp_path: Path) -> None:
        file_path = self._write(
            tmp_path / 'invalid-types.jsonl',
            '{"a": 1}\n\nnull\n[]\n"text"\n42\n{"a": 2}\n',
        )

        records = jsonl_to_list(file_path, skip_invalid=True)

        assert records == [{'a': 1}, {'a': 2}]

    def test_default_read_remains_strict_for_malformed_json(self, tmp_path: Path) -> None:
        file_path = self._write(tmp_path / 'strict.jsonl', '{"a": 1}\n{"a": 2')

        with pytest.raises(json.JSONDecodeError):
            jsonl_to_list(file_path)
