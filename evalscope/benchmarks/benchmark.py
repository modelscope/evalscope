# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path
from modelscope.msdatasets import MsDataset
from typing import Optional

from evalscope.constants import DEFAULT_DATASET_CACHE_DIR, HubType


class Benchmark(object):
    """
    Wrapper for loading datasets from ModelScope or HuggingFace.
    """

    def __init__(self):
        ...

    @staticmethod
    def load(dataset_name: str,
             subset: str = None,
             split: str = None,
             token: str = None,
             hub: str = 'ModelScope',
             work_dir: Optional[str] = DEFAULT_DATASET_CACHE_DIR,
             **kwargs):
        """
        Load a dataset from ModelScope or HuggingFace.

        Args:
            dataset_name (str): The dataset id or path.
                If it is dataset id, should be in the format of `organization/name` for ModelScope and HuggingFace hub.
                If it is dataset path, should be the path on local disk.
            subset (str):
            split:
            token: sdk token for ModelScope, optional, default None
            hub: `ModelScope` or `HuggingFace`
            work_dir: the work directory for caching, optional

        Returns:
            A dict.
        """

        dataset = MsDataset.load(
            dataset_name=dataset_name,
            subset_name=subset,
            split=split,
            token=token,
            cache_dir=work_dir,
            hub=hub,
            **kwargs)

        dataset.dataset_name = dataset_name.split('/')[-1]
        dataset.subset_name = subset
        # dataset.split = split
        return dataset


if __name__ == '__main__':

    ds = Benchmark.load(dataset_name='mmlu', subset='management', split=None)

    n = 1
    for i in ds:
        print('>', n, ': ', i)
        n += 1
