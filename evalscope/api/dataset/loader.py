import copy
import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from evalscope.api.dataset.utils import record_to_sample_fn
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, HubType
from evalscope.utils import get_logger
from evalscope.utils.io_utils import csv_to_list, gen_hash, jsonl_to_list, safe_filename, tsv_to_list
from .dataset import Dataset, FieldSpec, MemoryDataset, Sample
from .utils import data_to_samples, shuffle_choices_if_requested

logger = get_logger()


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    def __init__(
        self,
        data_id_or_path: str,
        split: str,
        sample_fields: Union[FieldSpec, Callable] = None,
        filter_func: Callable = None,
        subset: str = 'default',
        version: str = None,
        limit: Union[int, float] = None,
        data_source: Optional[str] = HubType.MODELSCOPE,
        shuffle: bool = False,
        shuffle_choices: Optional[Union[bool, int]] = None,
        seed: Optional[int] = None,
        auto_id: bool = True,
        repeats: int = 1,
        trust_remote: bool = True,
        force_redownload: bool = False,
        dataset_dir: Optional[str] = None,
        **kwargs
    ):
        self.data_id_or_path = data_id_or_path
        self.split = split
        self.sample_fields = sample_fields
        self.filter_func = filter_func
        self.subset = subset
        self.version = version
        self.limit = limit
        self.data_source = data_source
        self.shuffle = shuffle
        self.shuffle_choices = shuffle_choices
        self.seed = seed
        self.auto_id = auto_id
        self.repeats = repeats
        self.trust_remote = trust_remote
        self.force_redownload = force_redownload
        self.dataset_dir = dataset_dir
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> Dataset:
        """
        Load data from the source.
        """
        ...


class RemoteDataLoader(DataLoader):
    """
    Data loader for remote datasets: ModelScope or Huggingface.
    """

    def load(self) -> Dataset:
        import datasets
        from datasets import DownloadMode as HFDownloadMode
        from modelscope import MsDataset
        from modelscope.utils.constant import DownloadMode as MSDownloadMode

        path = self.data_id_or_path
        # resolve data_to_sample function
        data_to_sample = record_to_sample_fn(self.sample_fields)
        # generate a unique cache dir for this dataset
        dataset_hash = gen_hash(f'{path}{self.split}{self.subset}{self.version}{self.kwargs}')
        if self.dataset_dir:
            datasets_cache_dir = os.path.join(self.dataset_dir, 'datasets')
        else:
            datasets_cache_dir = os.path.join(DEFAULT_EVALSCOPE_CACHE_DIR, 'datasets')
        dataset_cache_dir = os.path.join(datasets_cache_dir, f'{safe_filename(path)}-{dataset_hash}')
        # force re-download: remove local cache if requested
        if self.force_redownload and os.path.exists(dataset_cache_dir):
            logger.info(f'Force redownload enabled. Removing cached dataset at: {dataset_cache_dir}')
            shutil.rmtree(dataset_cache_dir, ignore_errors=True)

        if os.path.exists(dataset_cache_dir):
            dataset = datasets.load_from_disk(dataset_cache_dir)
        else:
            logger.info(
                f'Loading dataset {path} from {self.data_source} > subset: {self.subset} > split: {self.split} ...'
            )
            # prepare download_mode for both backends when force_redownload is requested
            hf_download_mode = None if not self.force_redownload else HFDownloadMode.FORCE_REDOWNLOAD
            ms_download_mode = None if not self.force_redownload else MSDownloadMode.FORCE_REDOWNLOAD

            if self.data_source == HubType.MODELSCOPE:
                dataset = MsDataset.load(
                    dataset_name=path,
                    split=self.split,
                    subset_name=self.subset,
                    version=self.version,
                    trust_remote_code=self.trust_remote,
                    download_mode=ms_download_mode,
                    **self.kwargs,
                )
                # convert to Huggingface dataset if necessary
                if not isinstance(dataset, datasets.Dataset):
                    dataset = dataset.to_hf_dataset()
            elif self.data_source in [HubType.HUGGINGFACE, HubType.LOCAL]:
                # remove dataset_infos.json file if exists, since datasets will occur an error if it exists.
                dataset_infos_path = os.path.join(path, 'dataset_infos.json')
                if os.path.exists(dataset_infos_path):
                    logger.info(f'Removing dataset_infos.json file at {dataset_infos_path} to avoid datasets errors.')
                    os.remove(dataset_infos_path)
                # load dataset from Huggingface or local path
                dataset = datasets.load_dataset(
                    path=path,
                    name=self.subset if self.subset != 'default' else None,
                    split=self.split,
                    revision=self.version,
                    trust_remote_code=self.trust_remote,
                    download_mode=hf_download_mode,
                    **self.kwargs,
                )

            # Only save to disk if not loading from local path
            if self.data_source != HubType.LOCAL:
                dataset.save_to_disk(dataset_cache_dir)

        # shuffle if requested
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed)

        # limit if requested
        if self.limit:
            if isinstance(self.limit, float):
                self.limit = int(len(dataset) * self.limit)
            elif isinstance(self.limit, int) and self.limit < 0:
                raise ValueError('Limit must be a non-negative integer or a float between 0 and 1.')
            if len(dataset) > self.limit:
                dataset = dataset.select(range(self.limit))

        # convert to list
        dataset = dataset.to_list()

        # repeat k times
        if self.repeats > 1:
            dataset = [copy.deepcopy(item) for item in dataset for _ in range(self.repeats)]

        # return the dataset
        memory_dataset = MemoryDataset(
            samples=data_to_samples(data=dataset, data_to_sample=data_to_sample),
            name=Path(path).stem if Path(path).exists() else path,
            location=path,
        )

        # Apply filtering if a filter function is provided
        if self.filter_func is not None:
            memory_dataset = memory_dataset.filter(self.filter_func)

        # assign ids and group_ids if requested
        if self.auto_id:
            memory_dataset.reindex(group_size=self.repeats)

        shuffle_choices_if_requested(memory_dataset, self.shuffle_choices)

        return memory_dataset


class LocalDataLoader(DataLoader):
    """
    Data loader for local datasets. Reads from JSONL or CSV files.
    """

    def load(self):

        path = self.data_id_or_path
        data_to_sample = record_to_sample_fn(self.sample_fields)
        dataset = []

        # Check for JSONL or CSV files in the specified path
        for ext, loader in [
            ('.jsonl', jsonl_to_list),
            ('.csv', csv_to_list),
            ('.tsv', tsv_to_list),
        ]:
            # Check if the file exists with the given extension
            if os.path.isfile(path) and path.endswith(ext):
                file_paths = [path]
            else:
                file_paths = [
                    os.path.join(path, f'{self.subset}_{self.split}{ext}'),
                    os.path.join(path, f'{self.subset}{ext}')
                ]
            # If the file exists, load it
            for file_path in file_paths:
                if os.path.exists(file_path):
                    dataset = loader(file_path)
                    break  # Stop checking other extensions once a file is found

        # shuffle if requested
        if self.shuffle:
            random.shuffle(dataset, self.seed)

        # limit if requested
        if self.limit:
            if isinstance(self.limit, float):
                self.limit = int(len(dataset) * self.limit)
            elif isinstance(self.limit, int) and self.limit < 0:
                raise ValueError('Limit must be a non-negative integer or a float between 0 and 1.')
            dataset = dataset[:self.limit]

        # repeat k times
        if self.repeats > 1:
            dataset = [copy.deepcopy(item) for item in dataset for _ in range(self.repeats)]

        # return the dataset
        memory_dataset = MemoryDataset(
            samples=data_to_samples(data=dataset, data_to_sample=data_to_sample),
            name=Path(path).stem if Path(path).exists() else path,
            location=path,
        )

        # Apply filtering if a filter function is provided
        if self.filter_func is not None:
            memory_dataset = memory_dataset.filter(self.filter_func)

        # assign ids and group_ids if requested
        if self.auto_id:
            memory_dataset.reindex(group_size=self.repeats)

        shuffle_choices_if_requested(memory_dataset, self.shuffle_choices)

        return memory_dataset


class DictDataLoader(DataLoader):
    """Load dataset from a list of dictionaries."""

    def __init__(self, dict_list: list, **kwargs):
        super().__init__(data_id_or_path='', split='', **kwargs)
        self.dict_list = dict_list

    def load(self) -> Dataset:
        data_to_sample = record_to_sample_fn(self.sample_fields)
        dataset = self.dict_list

        # shuffle if requested
        if self.shuffle:
            random.shuffle(dataset, self.seed)

        # limit if requested
        if self.limit:
            if isinstance(self.limit, float):
                self.limit = int(len(dataset) * self.limit)
            elif isinstance(self.limit, int) and self.limit < 0:
                raise ValueError('Limit must be a non-negative integer or a float between 0 and 1.')
            dataset = dataset[:self.limit]

        # repeat k times
        if self.repeats > 1:
            dataset = [copy.deepcopy(item) for item in dataset for _ in range(self.repeats)]

        # return the dataset
        memory_dataset = MemoryDataset(samples=data_to_samples(data=dataset, data_to_sample=data_to_sample), )

        # Apply filtering if a filter function is provided
        if self.filter_func is not None:
            memory_dataset = memory_dataset.filter(self.filter_func)

        # assign ids and group_ids if requested
        if self.auto_id:
            memory_dataset.reindex(group_size=self.repeats)

        shuffle_choices_if_requested(memory_dataset, self.shuffle_choices)

        return memory_dataset
