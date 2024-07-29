# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path
from typing import Optional

from evalscope.constants import DEFAULT_ROOT_CACHE_DIR


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
             work_dir: Optional[str] = DEFAULT_ROOT_CACHE_DIR,
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
        work_dir = os.path.join(work_dir, 'benchmarks', dataset_name.replace('/', '_'))
        if hub == 'ModelScope':
            from modelscope.msdatasets import MsDataset
            dataset = MsDataset.load(dataset_name=dataset_name, subset_name=subset, split=split, token=token,
                                     cache_dir=work_dir, **kwargs)

            dataset.dataset_name = dataset_name.split('/')[-1]
            dataset.subset_name = subset
            dataset.split = split
            return dataset
        elif hub == 'HuggingFace':
            # TODO: implement this by xingjun.wxj@alibaba-inc.com
            raise NotImplementedError('HuggingFace hub is not supported yet.')
        else:
            raise ValueError(f'hub must be `ModelScope` or `HuggingFace`, but got {hub}')


if __name__ == '__main__':

    ds = Benchmark.load(dataset_name='mmlu', subset='management', split=None)

    n = 1
    for i in ds:
        print('>', n, ': ', i)
        n += 1
