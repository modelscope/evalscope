import copy
import errno
import json
import os
import random
import shutil
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Union
from uuid import uuid4

from filelock import FileLock

from evalscope.api.dataset.utils import record_to_sample_fn
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, HubType
from evalscope.utils import get_logger
from evalscope.utils.io_utils import (
    csv_to_list,
    gen_hash,
    jsonl_to_list,
    parquet_to_list,
    safe_filename,
    tsv_to_list,
    undecode_media,
)

from .dataset import Dataset, FieldSpec, MemoryDataset, Sample, resolve_dataset_limit, validate_dataset_limit
from .hub import DatasetHub
from .utils import data_to_samples, shuffle_choices_if_requested

logger = get_logger()

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset


class _DatasetLoadingSession:
    def __init__(self, stack: ExitStack) -> None:
        self.stack = stack
        self.directories: Dict[str, str] = {}

    def snapshot_cache_dir(self, cache_root: str) -> str:
        if cache_root not in self.directories:
            os.makedirs(cache_root, exist_ok=True)
            self.directories[cache_root] = self.stack.enter_context(
                TemporaryDirectory(prefix='.hf-staging-', dir=cache_root)
            )
        return self.directories[cache_root]

    def media_cache_file(self, cache_root: str) -> str:
        return os.path.join(self.snapshot_cache_dir(cache_root), f'media-{uuid4().hex}.arrow')


_loading_session: ContextVar[Optional[_DatasetLoadingSession]] = ContextVar('dataset_loading_session', default=None)


@contextmanager
def dataset_loading_session() -> Iterator[_DatasetLoadingSession]:
    """Share temporary snapshot parsing files until all benchmark splits have loaded."""
    existing = _loading_session.get()
    if existing is not None:
        yield existing
        return
    with ExitStack() as stack:
        session = _DatasetLoadingSession(stack)
        token = _loading_session.set(session)
        try:
            yield session
        finally:
            _loading_session.reset(token)


def _remove_dataset_cache(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.unlink(path)


def _recover_dataset_cache(cache_dir: str) -> None:
    """Recover an interrupted publication while holding the split lock."""
    import datasets

    backup = f'{cache_dir}.previous'
    if os.path.exists(backup):
        if os.path.exists(cache_dir):
            # A committed cache must be readable before its previous generation is discarded.
            with datasets.load_from_disk(cache_dir):
                pass
            _remove_dataset_cache(backup)
        else:
            os.replace(backup, cache_dir)
            logger.info(f'Restored dataset cache after interrupted publication: {cache_dir}')
    _remove_dataset_cache(f'{cache_dir}.incomplete')


def _publish_dataset_cache(dataset: 'HFDataset', cache_dir: str) -> None:
    """Publish a validated split while holding the split lock."""
    import datasets

    prepared = f'{cache_dir}.incomplete'
    backup = f'{cache_dir}.previous'
    try:
        dataset.save_to_disk(prepared)
        with datasets.load_from_disk(prepared) as saved:
            if saved.num_rows != dataset.num_rows or saved.features != dataset.features:
                raise ValueError(f'Saved dataset does not match its source: {cache_dir}')
        had_previous = os.path.exists(cache_dir)
        try:
            if had_previous:
                os.replace(cache_dir, backup)
            os.replace(prepared, cache_dir)
        except BaseException:
            if os.path.exists(backup) and not os.path.exists(cache_dir):
                os.replace(backup, cache_dir)
            raise
        if had_previous:
            _remove_dataset_cache(backup)
    finally:
        _remove_dataset_cache(prepared)


def _shuffle_in_place(data: list, seed: Optional[int]) -> None:
    """Shuffle a list in place with an optional seed.

    Uses random.Random(seed) instead of the deprecated random.shuffle(x, random)
    which was removed in Python 3.12.
    """
    if seed is not None:
        random.Random(seed).shuffle(data)
    else:
        random.shuffle(data)


def _dataset_cache_hash(
    data_id_or_path: str,
    split: str,
    subset: str,
    version: Optional[str],
    data_source: Optional[str],
    kwargs: Dict,
) -> str:
    """Build a stable hash from every input that determines a remote dataset."""
    effective_data_source = _resolve_effective_data_source(data_id_or_path, data_source)
    payload = {
        'data_id_or_path': data_id_or_path,
        'split': split,
        'subset': subset,
        'version': version,
        'data_source': effective_data_source,
        'kwargs': kwargs,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return gen_hash(serialized)


def _resolve_effective_data_source(data_id_or_path: str, data_source: Optional[str]) -> str:
    """Resolve the source using the same local-path precedence as DatasetHub."""
    if data_source == HubType.LOCAL or os.path.exists(data_id_or_path):
        return HubType.LOCAL
    return data_source or HubType.MODELSCOPE


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
        **kwargs,
    ):
        self.data_id_or_path = data_id_or_path
        self.split = split
        self.sample_fields = sample_fields
        self.filter_func = filter_func
        self.subset = subset
        self.version = version
        self.limit = validate_dataset_limit(limit)
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

        path = self.data_id_or_path
        effective_data_source = _resolve_effective_data_source(path, self.data_source)
        # generate a unique cache dir for this dataset
        dataset_hash = _dataset_cache_hash(
            path,
            self.split,
            self.subset,
            self.version,
            self.data_source,
            self.kwargs,
        )
        if self.dataset_dir:
            datasets_cache_dir = os.path.join(self.dataset_dir, 'datasets')
        else:
            datasets_cache_dir = os.path.join(DEFAULT_EVALSCOPE_CACHE_DIR, 'datasets')
        dataset_cache_dir = os.path.join(datasets_cache_dir, f'{safe_filename(path)}-{dataset_hash}')
        hub = DatasetHub(
            data_id_or_path=path,
            data_source=effective_data_source,
            revision=self.version,
            trust_remote=self.trust_remote,
            force_redownload=self.force_redownload,
        )
        if effective_data_source == HubType.LOCAL:
            with hub.load(split=self.split, subset=self.subset, **self.kwargs) as dataset:
                return self._to_memory_dataset(dataset)

        os.makedirs(datasets_cache_dir, exist_ok=True)
        # Readers hold the same lock through conversion so refresh cannot remove their mapped files.
        with dataset_loading_session() as session:
            try:
                lock = FileLock(f'{dataset_cache_dir}.lock').acquire()
            except OSError as exc:
                # Prebuilt read-only caches remain usable; they cannot be refreshed by this process.
                read_only_cache = dataset_cache_dir
                if not os.path.isdir(read_only_cache):
                    read_only_cache = f'{dataset_cache_dir}.previous'
                if (
                    exc.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}
                    or os.access(datasets_cache_dir, os.W_OK)
                    or self.force_redownload
                    or not os.path.isdir(read_only_cache)
                ):
                    raise
                with datasets.load_from_disk(read_only_cache) as dataset:
                    return self._to_memory_dataset(dataset)
            with lock:
                return self._load_cached_split(hub, session, dataset_cache_dir)

    def _load_cached_split(
        self,
        hub: DatasetHub,
        session: _DatasetLoadingSession,
        cache_dir: str,
    ) -> MemoryDataset:
        import datasets

        def media_cache_file() -> str:
            return session.media_cache_file(os.path.dirname(cache_dir))

        _recover_dataset_cache(cache_dir)
        if os.path.exists(cache_dir) and not self.force_redownload:
            try:
                dataset = datasets.load_from_disk(cache_dir)
            except FileNotFoundError:
                logger.info(f'Incomplete dataset cache at {cache_dir}; rebuilding it.')
            else:
                with dataset:
                    return self._to_memory_dataset(dataset, media_cache_file)
        logger.info(
            f'Loading dataset {self.data_id_or_path} from {hub.data_source} '
            f'> subset: {self.subset} > split: {self.split} ...'
        )
        with hub.load(
            split=self.split,
            subset=self.subset,
            _snapshot_cache_dir=lambda: session.snapshot_cache_dir(os.path.dirname(cache_dir)),
            **self.kwargs,
        ) as dataset:
            _publish_dataset_cache(dataset, cache_dir)
        with datasets.load_from_disk(cache_dir) as dataset:
            return self._to_memory_dataset(dataset, media_cache_file)

    def _to_memory_dataset(
        self,
        dataset: 'HFDataset',
        media_cache_file: Optional[Callable[[], str]] = None,
    ) -> MemoryDataset:
        data_to_sample = record_to_sample_fn(self.sample_fields)
        path = self.data_id_or_path

        # Disable auto-decoding for media columns to keep their raw bytes representation.
        dataset = undecode_media(
            dataset, media_type=['image', 'audio', 'video'], cache_file_name_factory=media_cache_file
        )

        # shuffle if requested
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed)

        # limit if requested
        if self.limit:
            resolved_limit = resolve_dataset_limit(self.limit, len(dataset))
            if resolved_limit is not None and len(dataset) > resolved_limit:
                dataset = dataset.select(range(resolved_limit))

        # convert to list
        dataset_list = list(dataset)

        # repeat k times
        if self.repeats > 1:
            dataset_list = [copy.deepcopy(item) for item in dataset_list for _ in range(self.repeats)]

        # return the dataset
        memory_dataset = MemoryDataset(
            samples=data_to_samples(data=dataset_list, data_to_sample=data_to_sample),
            name=Path(path).stem if Path(path).exists() else path,
            location=path,
        )

        # Apply filtering if a filter function is provided
        if self.filter_func is not None:
            memory_dataset = memory_dataset.filter(self.filter_func)

        # assign ids and group_ids if requested
        if self.auto_id:
            memory_dataset.reindex(group_size=self.repeats)

        shuffle_choices_if_requested(memory_dataset, self.shuffle_choices, self.seed)

        return memory_dataset


class LocalDataLoader(DataLoader):
    """
    Data loader for local datasets. Reads from JSONL or CSV files.
    """

    def load(self):

        path = self.data_id_or_path
        data_to_sample = record_to_sample_fn(self.sample_fields)
        dataset = []
        supported_format = [
            ('.jsonl', jsonl_to_list),
            ('.csv', csv_to_list),
            ('.tsv', tsv_to_list),
            ('.parquet', parquet_to_list),
        ]

        dataset_found = False

        # Check for JSONL or CSV files in the specified path
        for ext, loader in supported_format:
            if dataset_found:
                break

            # Check if the file exists with the given extension
            if os.path.isfile(path) and path.endswith(ext):
                file_paths = [path]
            else:
                file_paths = [
                    os.path.join(path, f'{self.subset}_{self.split}{ext}'),
                    os.path.join(path, f'{self.subset}{ext}'),
                ]
            # If the file exists, load it
            for file_path in file_paths:
                if os.path.exists(file_path):
                    dataset = loader(file_path)
                    if dataset:
                        dataset_found = True
                    break

        # If no specific file found, raise an error with helpful information
        if not dataset_found:
            supported_exts = [ext for ext, _ in supported_format]
            if os.path.isdir(path):
                # Directory path: list expected candidates and available files for diagnosis
                expected_with_split = [os.path.join(path, f'{self.subset}_{self.split}{ext}') for ext in supported_exts]
                available_files = sorted([f for f in os.listdir(path) if os.path.splitext(f)[1] in supported_exts])
                raise FileNotFoundError(
                    f'No dataset file found for subset="{self.subset}", split="{self.split}" in "{path}".\n'
                    f'Expected one of:\n'
                    + '\n'.join(f'  - {p}' for p in expected_with_split)
                    + '\n'
                    + 'Available files in "'
                    + path
                    + '":\n'
                    + ('\n'.join(f'  - {f}' for f in available_files) if available_files else '  (none)')
                )
            elif os.path.isfile(path):
                # Direct file path provided but unsupported extension
                _, file_ext = os.path.splitext(path)
                if file_ext not in supported_exts:
                    raise FileNotFoundError(
                        f'Unsupported file format "{file_ext}" for "{path}". Supported formats: {supported_exts}'
                    )
            else:
                raise FileNotFoundError(f'Dataset path does not exist: "{path}"')

        # shuffle if requested
        if self.shuffle:
            _shuffle_in_place(dataset, self.seed)

        # limit if requested
        if self.limit:
            resolved_limit = resolve_dataset_limit(self.limit, len(dataset))
            dataset = dataset[:resolved_limit]

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

        shuffle_choices_if_requested(memory_dataset, self.shuffle_choices, self.seed)

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
            _shuffle_in_place(dataset, self.seed)

        # limit if requested
        if self.limit:
            resolved_limit = resolve_dataset_limit(self.limit, len(dataset))
            dataset = dataset[:resolved_limit]

        # repeat k times
        if self.repeats > 1:
            dataset = [copy.deepcopy(item) for item in dataset for _ in range(self.repeats)]

        # return the dataset
        memory_dataset = MemoryDataset(
            samples=data_to_samples(data=dataset, data_to_sample=data_to_sample),
        )

        # Apply filtering if a filter function is provided
        if self.filter_func is not None:
            memory_dataset = memory_dataset.filter(self.filter_func)

        # assign ids and group_ids if requested
        if self.auto_id:
            memory_dataset.reindex(group_size=self.repeats)

        shuffle_choices_if_requested(memory_dataset, self.shuffle_choices, self.seed)

        return memory_dataset
