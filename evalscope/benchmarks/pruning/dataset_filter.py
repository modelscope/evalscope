from __future__ import annotations

from typing import Set

from evalscope.api.dataset import DatasetDict, MemoryDataset


def filter_dataset_by_indices(dataset: DatasetDict, keep_indices: Set[int]) -> DatasetDict:
    pruned = {}

    for subset_name, subset_data in dataset.items():
        selected = []
        for position, sample in enumerate(subset_data):
            sample_index = sample.id if sample.id is not None else position
            if int(sample_index) in keep_indices:
                selected.append(sample)

        pruned_subset = MemoryDataset(
            samples=selected,
            name=getattr(subset_data, "name", subset_name),
            location=getattr(subset_data, "location", None),
            shuffled=False,
        )
        pruned_subset.reindex()
        pruned[subset_name] = pruned_subset

    return DatasetDict(pruned)


def filter_dataset_by_metadata_id(dataset: DatasetDict, keep_ids: Set[str]) -> DatasetDict:
    pruned = {}

    for subset_name, subset_data in dataset.items():
        selected = []
        for sample in subset_data:
            sample_id = str((sample.metadata or {}).get("id", ""))
            if sample_id in keep_ids:
                selected.append(sample)

        if selected:
            pruned_subset = MemoryDataset(
                samples=selected,
                name=getattr(subset_data, "name", subset_name),
                location=getattr(subset_data, "location", None),
                shuffled=False,
            )
            pruned_subset.reindex()
            pruned[subset_name] = pruned_subset

    return DatasetDict(pruned)
