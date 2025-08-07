import copy
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from evalscope.api.dataset.utils import record_to_sample_fn
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, HubType
from evalscope.utils import gen_hash, get_logger, safe_filename
from .dataset import Dataset, FieldSpec, MemoryDataset, Sample
from .utils import data_to_samples

logger = get_logger()


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    def __init__(self,
                 data_id_or_path: str,
                 split: str,
                 sample_fields: Union[FieldSpec, Callable] = None,
                 subset: str = 'default',
                 version: str = None,
                 limit: Union[int, float] = None,
                 data_source: Optional[str] = None,
                 shuffle: bool = False,
                 seed: Optional[int] = None,
                 auto_id: bool = True,
                 repeats: int = 1,
                 trust_remote: bool = True,
                 **kwargs):
        self.data_id_or_path = data_id_or_path
        self.split = split
        self.sample_fields = sample_fields
        self.subset = subset
        self.version = version
        self.limit = limit
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.auto_id = auto_id
        self.repeats = repeats
        self.trust_remote = trust_remote
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
        from modelscope import MsDataset

        path = self.data_id_or_path
        # resolve data_to_sample function
        data_to_sample = record_to_sample_fn(self.sample_fields)
        # generate a unique cache dir for this dataset
        dataset_hash = gen_hash(f'{path}{self.split}{self.subset}{self.version}{self.kwargs}')
        datasets_cache_dir = os.path.join(DEFAULT_EVALSCOPE_CACHE_DIR, 'datasets')
        dataset_cache_dir = os.path.join(datasets_cache_dir, f'{safe_filename(path)}-{dataset_hash}')
        if os.path.exists(dataset_cache_dir):
            dataset = datasets.load_from_disk(dataset_cache_dir)
        else:
            logger.info(f'Loading dataset {path} from {self.data_source}...')
            if self.data_source == HubType.MODELSCOPE:
                dataset = MsDataset.load(
                    dataset_name=path,
                    split=self.split,
                    subset_name=self.subset,
                    version=self.version,
                    trust_remote_code=self.trust_remote,
                )
                if not isinstance(dataset, datasets.Dataset):
                    dataset = dataset.to_hf_dataset()
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
            dataset = dataset.select(range(self.limit))

        # convert to list
        dataset = dataset.to_list()

        # repeat k times
        if self.repeats > 1:
            dataset = [copy.deepcopy(item) for item in dataset for _ in range(self.repeats)]

        # return the dataset
        memory_dataset = MemoryDataset(
            samples=data_to_samples(
                data=dataset, data_to_sample=data_to_sample, auto_id=self.auto_id, group_k=self.repeats),
            name=Path(path).stem if Path(path).exists() else path,
            location=path,
        )

        return memory_dataset
