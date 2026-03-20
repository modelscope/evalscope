# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for dataset shuffle compatibility with Python 3.12+.

Python 3.12 removed the deprecated second argument from random.shuffle().
LocalDataLoader and DictDataLoader must use random.Random(seed).shuffle()
for reproducible shuffling instead of random.shuffle(dataset, seed).
"""

import random
import sys
import unittest
from unittest.mock import MagicMock, patch

from evalscope.api.dataset.loader import DictDataLoader


class TestShuffleCompat(unittest.TestCase):
    """random.shuffle(x, seed) must not be called on Python 3.12+."""

    def _make_dict_list(self, n=20):
        return [{'input': str(i), 'ideal': str(i)} for i in range(n)]

    @patch('evalscope.api.dataset.loader.shuffle_choices_if_requested')
    @patch('evalscope.api.dataset.loader.MemoryDataset')
    @patch('evalscope.api.dataset.loader.data_to_samples',
           side_effect=lambda **kw: kw['data'])
    def test_shuffle_with_seed_does_not_crash(self, _d2s, mock_ds, _sc):
        """DictDataLoader.load() with shuffle=True, seed=42 must not raise TypeError."""
        mock_ds.return_value = MagicMock()
        data = self._make_dict_list()
        loader = DictDataLoader(dict_list=list(data), shuffle=True, seed=42)
        loader.load()

    @patch('evalscope.api.dataset.loader.shuffle_choices_if_requested')
    @patch('evalscope.api.dataset.loader.MemoryDataset')
    @patch('evalscope.api.dataset.loader.data_to_samples',
           side_effect=lambda **kw: kw['data'])
    def test_shuffle_without_seed_does_not_crash(self, _d2s, mock_ds, _sc):
        """DictDataLoader.load() with shuffle=True, seed=None must not raise TypeError."""
        mock_ds.return_value = MagicMock()
        data = self._make_dict_list()
        loader = DictDataLoader(dict_list=list(data), shuffle=True, seed=None)
        loader.load()

    @patch('evalscope.api.dataset.loader.shuffle_choices_if_requested')
    @patch('evalscope.api.dataset.loader.MemoryDataset')
    @patch('evalscope.api.dataset.loader.data_to_samples',
           side_effect=lambda **kw: kw['data'])
    def test_seeded_shuffle_is_reproducible(self, _d2s, mock_ds, _sc):
        """Same seed must produce the same shuffled order."""
        mock_ds.return_value = MagicMock()
        data_a = self._make_dict_list()
        data_b = self._make_dict_list()

        DictDataLoader(dict_list=data_a, shuffle=True, seed=123).load()
        DictDataLoader(dict_list=data_b, shuffle=True, seed=123).load()

        self.assertEqual(data_a, data_b)
        self.assertNotEqual(data_a, self._make_dict_list())

    def test_original_call_fails_on_312(self):
        """Confirm random.shuffle(x, seed) raises TypeError on Python 3.11+."""
        if sys.version_info < (3, 11):
            self.skipTest('Only fails on Python 3.11+')
        with self.assertRaises(TypeError):
            random.shuffle([1, 2, 3], 42)


if __name__ == '__main__':
    unittest.main()
