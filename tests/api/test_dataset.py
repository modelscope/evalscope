import unittest

from evalscope.api.dataset.dataset import DatasetDict, MemoryDataset, Sample

SUBSET = 'default'


def _make_dataset(n: int, name: str = 'test') -> MemoryDataset:
    """Create a MemoryDataset with *n* distinct samples (input = 'question_i')."""
    samples = [
        Sample(input=f'question_{i}', target=f'answer_{i}', subset_key=SUBSET)
        for i in range(n)
    ]
    return MemoryDataset(samples, name=name)


class TestDatasetDictFromDatasetRepeats(unittest.TestCase):
    """Unit tests for DatasetDict.from_dataset — repeats parameter."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_samples(self, dataset_dict: DatasetDict):
        """Return the flat sample list for the single subset."""
        return list(dataset_dict[SUBSET])

    # ------------------------------------------------------------------
    # core regression test (PR #1363)
    # ------------------------------------------------------------------

    def test_repeats_produces_copies_not_distinct_samples(self):
        """limit=5, repeats=3 must yield [s0,s0,s0, s1,s1,s1, …], not 15 distinct samples."""
        dataset = _make_dataset(20)
        result = DatasetDict.from_dataset(dataset, subset_list=[SUBSET], limit=5, repeats=3)
        samples = self._get_samples(result)

        # Total count: 5 * 3 = 15
        self.assertEqual(len(samples), 15, 'Expected 5 original samples × 3 repeats = 15 total')

        # Each trio should share the same original input (not 15 distinct inputs)
        for i in range(5):
            base_input = samples[i * 3].input
            self.assertEqual(samples[i * 3 + 1].input, base_input,
                             f'group {i}: copy 1 input differs from copy 0')
            self.assertEqual(samples[i * 3 + 2].input, base_input,
                             f'group {i}: copy 2 input differs from copy 0')

        # Verify the first 5 original inputs are used (not inputs 5-19)
        expected_inputs = [f'question_{i}' for i in range(5)]
        actual_inputs = [samples[i * 3].input for i in range(5)]
        self.assertEqual(actual_inputs, expected_inputs,
                         'The first `limit` samples should be used as the source, not samples beyond limit')

    # ------------------------------------------------------------------
    # group_id correctness
    # ------------------------------------------------------------------

    def test_group_id_assigned_correctly(self):
        """Each group of k copies must share the same group_id."""
        dataset = _make_dataset(10)
        result = DatasetDict.from_dataset(dataset, subset_list=[SUBSET], limit=4, repeats=3)
        samples = self._get_samples(result)

        # 4 groups × 3 copies = 12 samples
        self.assertEqual(len(samples), 12)

        for i in range(4):
            expected_group_id = i
            for j in range(3):
                idx = i * 3 + j
                self.assertEqual(
                    samples[idx].group_id, expected_group_id,
                    f'sample index {idx}: expected group_id={expected_group_id}, got {samples[idx].group_id}'
                )

    def test_id_is_globally_unique(self):
        """After reindex every sample must have a unique id."""
        dataset = _make_dataset(10)
        result = DatasetDict.from_dataset(dataset, subset_list=[SUBSET], limit=4, repeats=3)
        samples = self._get_samples(result)
        ids = [s.id for s in samples]
        self.assertEqual(len(ids), len(set(ids)), 'Sample ids must be unique after reindex')

    # ------------------------------------------------------------------
    # deep-copy isolation
    # ------------------------------------------------------------------

    def test_copies_are_independent_objects(self):
        """The k copies within each group must be distinct objects (deepcopy)."""
        dataset = _make_dataset(5)
        result = DatasetDict.from_dataset(dataset, subset_list=[SUBSET], limit=3, repeats=2)
        samples = self._get_samples(result)

        for i in range(3):
            copy_a = samples[i * 2]
            copy_b = samples[i * 2 + 1]
            self.assertIsNot(copy_a, copy_b,
                             f'group {i}: the two copies must be different objects (deepcopy required)')

    # ------------------------------------------------------------------
    # boundary: repeats=1 (no duplication)
    # ------------------------------------------------------------------

    def test_repeats_one_returns_original_samples(self):
        """repeats=1 must return exactly `limit` samples with no duplication."""
        dataset = _make_dataset(10)
        result = DatasetDict.from_dataset(dataset, subset_list=[SUBSET], limit=5, repeats=1)
        samples = self._get_samples(result)

        self.assertEqual(len(samples), 5)
        for i, s in enumerate(samples):
            self.assertEqual(s.input, f'question_{i}')

    # ------------------------------------------------------------------
    # boundary: limit=None with repeats > 1
    # ------------------------------------------------------------------

    def test_repeats_without_limit(self):
        """limit=None with repeats=2 must repeat every sample in the full dataset."""
        dataset = _make_dataset(4)
        result = DatasetDict.from_dataset(dataset, subset_list=[SUBSET], limit=None, repeats=2)
        samples = self._get_samples(result)

        self.assertEqual(len(samples), 8, 'All 4 samples × 2 repeats = 8')
        for i in range(4):
            self.assertEqual(samples[i * 2].input, f'question_{i}')
            self.assertEqual(samples[i * 2 + 1].input, f'question_{i}')

    # ------------------------------------------------------------------
    # boundary: float limit
    # ------------------------------------------------------------------

    def test_float_limit_with_repeats(self):
        """float limit=0.5 on a 10-sample dataset → 5 samples, then repeated k times."""
        dataset = _make_dataset(10)
        result = DatasetDict.from_dataset(dataset, subset_list=[SUBSET], limit=0.5, repeats=3)
        samples = self._get_samples(result)

        self.assertEqual(len(samples), 15, 'float limit 0.5 × 10 = 5 samples × 3 repeats = 15')
        for i in range(5):
            base = samples[i * 3].input
            self.assertEqual(samples[i * 3 + 1].input, base)
            self.assertEqual(samples[i * 3 + 2].input, base)


if __name__ == '__main__':
    unittest.main()
