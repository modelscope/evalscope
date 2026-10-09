# Copyright (c) Alibaba, Inc. and its affiliates.

import inspect
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional, Sequence, Union

from evalscope.constants import HubType
from evalscope.utils.logger import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from datasets import DownloadConfig
    from datasets import IterableDataset as HFIterableDataset


# These SDK controls have no equivalent local Hugging Face loading semantics.
_MODELSCOPE_LOAD_OPTIONS = {
    'namespace',
    'target',
    'hub',
    'use_streaming',
    'stream_batch_size',
    'custom_cfg',
    'dataset_info_only',
    'download_mode',
    'engine',
}

# Include legacy options so older datasets versions do not pass them to the builder.
_HF_LOAD_ONLY_OPTIONS = {
    'split',
    'streaming',
    'num_proc',
    'keep_in_memory',
    'save_infos',
    'verification_mode',
    'ignore_verifications',
    'task',
}


@dataclass(frozen=True)
class DatasetHub:
    """Small hub handle shared by dataset loaders and benchmark-specific media resolvers."""

    data_id_or_path: str
    data_source: Optional[str] = HubType.MODELSCOPE
    revision: Optional[str] = None
    trust_remote: bool = True
    force_redownload: bool = False
    cache_dir: Optional[str] = None

    def load(
        self,
        split: str,
        subset: str = 'default',
        *,
        _snapshot_cache_dir: Optional[Callable[[], str]] = None,
        **kwargs: Any,
    ) -> 'Union[HFDataset, HFIterableDataset]':
        """Load a split, optionally using caller-owned storage for cached snapshot parsing."""
        source = _resolve_data_source(self.data_id_or_path, self.data_source)
        if source in (HubType.LOCAL, HubType.HUGGINGFACE):
            return self._load_huggingface(self.data_id_or_path, split, subset, kwargs)
        if source == HubType.MODELSCOPE:
            return self._load_modelscope(split, subset, kwargs, _snapshot_cache_dir)
        raise ValueError(f'Unsupported dataset hub: {source}')

    def _load_huggingface(
        self,
        path: str,
        split: str,
        subset: str,
        options: Mapping[str, Any],
        *,
        cached_snapshot: bool = False,
        snapshot_cache_dir: Optional[Callable[[], str]] = None,
    ) -> 'Union[HFDataset, HFIterableDataset]':
        import datasets

        # Explicit local mirrors retain the legacy metadata workaround; SDK snapshots remain untouched.
        dataset_infos_path = os.path.join(path, 'dataset_infos.json')
        if not cached_snapshot and os.path.exists(dataset_infos_path):
            logger.info(f'Removing dataset_infos.json file at {dataset_infos_path} to avoid datasets errors.')
            os.remove(dataset_infos_path)

        load_kwargs = dict(
            path=path,
            name=subset if subset != 'default' else None,
            split=split,
            revision=self.revision,
            download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD if self.force_redownload else None,
            **options,
        )
        if 'trust_remote_code' in inspect.signature(datasets.load_dataset).parameters:
            load_kwargs['trust_remote_code'] = self.trust_remote
        if cached_snapshot:
            load_kwargs = _prepare_cached_dataset_kwargs(load_kwargs, snapshot_cache_dir)
        return datasets.load_dataset(**load_kwargs)

    def _load_modelscope(
        self,
        split: str,
        subset: str,
        options: Mapping[str, Any],
        snapshot_cache_dir: Optional[Callable[[], str]],
    ) -> 'Union[HFDataset, HFIterableDataset]':
        """Try the SDK snapshot, falling back only when cached files or splits are missing."""
        import datasets

        snapshot = None
        if not self.force_redownload and not _MODELSCOPE_LOAD_OPTIONS.intersection(options):
            snapshot = _try_modelscope_cached_snapshot(self.data_id_or_path, revision=self.revision)
        if snapshot:
            try:
                return self._load_huggingface(
                    snapshot,
                    split,
                    subset,
                    options,
                    cached_snapshot=True,
                    snapshot_cache_dir=snapshot_cache_dir,
                )
            except (FileNotFoundError, ValueError) as exc:
                if not _is_missing_cached_data(exc):
                    raise
                logger.info(
                    f'Cached ModelScope dataset {self.data_id_or_path} cannot provide subset {subset}, '
                    f'split {split}: {exc}. Falling back to MsDataset.load.'
                )

        from modelscope import MsDataset
        from modelscope.utils.constant import DownloadMode

        load_kwargs = dict(
            dataset_name=self.data_id_or_path,
            split=split,
            subset_name=subset if subset != 'default' else None,
            trust_remote_code=self.trust_remote,
            **options,
        )
        if self.revision:
            load_kwargs['version'] = self.revision
        if self.force_redownload:
            load_kwargs['download_mode'] = DownloadMode.FORCE_REDOWNLOAD
        dataset = MsDataset.load(**load_kwargs)
        return dataset if isinstance(dataset, datasets.Dataset) else dataset.to_hf_dataset()

    def download_file(self, file_path: str) -> str:
        return download_dataset_file(
            data_id_or_path=self.data_id_or_path,
            file_path=file_path,
            data_source=self.data_source,
            revision=self.revision,
            force_redownload=self.force_redownload,
            cache_dir=self.cache_dir,
        )

    def download_snapshot(
        self,
        allow_file_pattern: Optional[Union[str, List[str]]] = None,
        ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    ) -> str:
        return download_dataset_snapshot(
            data_id_or_path=self.data_id_or_path,
            data_source=self.data_source,
            revision=self.revision,
            force_redownload=self.force_redownload,
            cache_dir=self.cache_dir,
            allow_file_pattern=allow_file_pattern,
            ignore_file_pattern=ignore_file_pattern,
        )


def _resolve_data_source(data_id_or_path: str, data_source: Optional[str]) -> str:
    if data_source == HubType.LOCAL or os.path.exists(data_id_or_path):
        return HubType.LOCAL
    return data_source or HubType.MODELSCOPE


def _try_modelscope_cached_snapshot(
    data_id_or_path: str,
    revision: Optional[str] = None,
) -> Optional[str]:
    """Resolve a ModelScope snapshot offline using the SDK's cache configuration."""
    download_kwargs = dict(repo_id=data_id_or_path, repo_type='dataset', local_files_only=True)
    if revision:
        download_kwargs['revision'] = revision
    try:
        from modelscope import snapshot_download

        snapshot_path = snapshot_download(**download_kwargs)
    except Exception as exc:
        logger.debug(f'No reusable ModelScope snapshot for {data_id_or_path}: {exc}')
        return None
    if not os.path.isdir(snapshot_path):
        return None
    # Legacy caches are unversioned; their offline API cannot verify a requested revision.
    if revision and not os.path.normpath(snapshot_path).endswith(os.path.join(os.sep, 'snapshots', revision)):
        logger.debug(f'Skipping ModelScope cache with an unverified revision: {snapshot_path}')
        return None
    logger.info(f'Using cached ModelScope dataset {data_id_or_path} from {snapshot_path}')
    return snapshot_path


def _validate_cached_shards(data_files: Mapping[str, Sequence[str]]) -> None:
    """Check completeness when numbered filenames declare a shard count."""
    for files in data_files.values():
        groups: dict[tuple[str, str, str], tuple[int, set[int]]] = {}
        for path in files:
            match = re.fullmatch(r'(.+)-(\d+)-of-(\d+)(\..+)', os.path.basename(path))
            if match is None:
                continue
            prefix, index_text, count_text, suffix = match.groups()
            index, count = int(index_text), int(count_text)
            if count < 1 or index >= count:
                raise ValueError(f'Invalid numbered shard: {path}')
            key = (os.path.dirname(path), prefix, suffix)
            expected, indices = groups.setdefault(key, (count, set()))
            if expected != count or index in indices:
                raise ValueError(f'Inconsistent numbered shards: {path}')
            indices.add(index)
        for (directory, prefix, suffix), (count, indices) in groups.items():
            if len(indices) != count:
                raise FileNotFoundError(
                    f'Incomplete cached shards for {os.path.join(directory, prefix)}*{suffix}: '
                    f'found {len(indices)} of {count}.'
                )


def _prepare_cached_dataset_kwargs(
    load_kwargs: Mapping[str, Any],
    snapshot_cache_dir: Optional[Callable[[], str]] = None,
) -> dict[str, Any]:
    """Resolve snapshot files for validation and a distinct Hugging Face cache identity."""
    import datasets

    builder_kwargs = {key: value for key, value in load_kwargs.items() if key not in _HF_LOAD_ONLY_OPTIONS}
    builder = datasets.load_dataset_builder(**builder_kwargs)
    prepared_kwargs = dict(load_kwargs)
    if not builder.config.data_files:
        return prepared_kwargs
    if load_kwargs.get('data_files') is None:
        _validate_cached_shards(builder.config.data_files)
    # Absolute files and their metadata distinguish snapshots in the HF cache key.
    prepared_kwargs['data_files'] = builder.config.data_files
    # Script loaders may create resources inside their builder output directory.
    # Only standard file builders use temporary Arrow storage; explicit cache settings remain authoritative.
    if snapshot_cache_dir is None or load_kwargs.get('cache_dir') is not None:
        return prepared_kwargs
    if not type(builder).__module__.startswith('datasets.packaged_modules.'):
        return prepared_kwargs
    prepared_kwargs['cache_dir'] = snapshot_cache_dir()
    prepared_kwargs['download_config'] = _persistent_download_config(load_kwargs)
    return prepared_kwargs


def _persistent_download_config(load_kwargs: Mapping[str, Any]) -> 'DownloadConfig':
    """Keep downloaded resources outside temporary Arrow storage without mutating caller options."""
    import datasets

    config = deepcopy(load_kwargs.get('download_config')) or datasets.DownloadConfig(
        num_proc=load_kwargs.get('num_proc'),
        token=load_kwargs.get('token'),
        storage_options=load_kwargs.get('storage_options') or {},
        use_etag=False,
    )
    if config.cache_dir is None:
        config.cache_dir = str(datasets.config.DOWNLOADED_DATASETS_PATH)
    return config


def _is_missing_cached_data(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if not isinstance(exc, ValueError):
        return False
    # datasets exposes missing splits through ValueError messages.
    message = str(exc)
    return (message.startswith('Unknown split "') and '. Should be one of ' in message) or (
        message.startswith('Bad split: ') and '. Available splits: ' in message
    )


def load_dataset_from_hub(
    data_id_or_path: str,
    split: str,
    subset: str = 'default',
    data_source: Optional[str] = HubType.MODELSCOPE,
    version: Optional[str] = None,
    trust_remote: bool = True,
    force_redownload: bool = False,
    _snapshot_cache_dir: Optional[Callable[[], str]] = None,
    **kwargs: Any,
) -> 'Union[HFDataset, HFIterableDataset]':
    """Load a dataset split from ModelScope, Hugging Face, or a local dataset path."""
    hub = DatasetHub(
        data_id_or_path=data_id_or_path,
        data_source=data_source,
        revision=version,
        trust_remote=trust_remote,
        force_redownload=force_redownload,
    )
    return hub.load(split=split, subset=subset, _snapshot_cache_dir=_snapshot_cache_dir, **kwargs)


def download_dataset_file(
    data_id_or_path: str,
    file_path: str,
    data_source: Optional[str] = HubType.MODELSCOPE,
    revision: Optional[str] = None,
    force_redownload: bool = False,
    cache_dir: Optional[str] = None,
) -> str:
    """Download or resolve a single file from a dataset hub."""
    data_source = _resolve_data_source(data_id_or_path, data_source)

    if data_source == HubType.LOCAL:
        root_dir = os.path.realpath(data_id_or_path)
        resolved_path = os.path.realpath(os.path.join(root_dir, file_path))
        if os.path.commonpath([root_dir, resolved_path]) != root_dir:
            raise ValueError(f'Invalid dataset file path: {file_path}')
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f'Dataset file {file_path} was not found in {root_dir}.')
        return resolved_path

    if data_source == HubType.HUGGINGFACE:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=data_id_or_path,
            filename=file_path,
            repo_type='dataset',
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_redownload,
        )

    if data_source == HubType.MODELSCOPE:
        from modelscope import dataset_file_download

        download_kwargs = {}
        if revision:
            download_kwargs['revision'] = revision
        if cache_dir:
            download_kwargs['cache_dir'] = cache_dir
        if not force_redownload:
            # Cache probe: any failure falls through to the real download below.
            try:
                return dataset_file_download(data_id_or_path, file_path, local_files_only=True, **download_kwargs)
            except Exception:
                pass
        return dataset_file_download(data_id_or_path, file_path, **download_kwargs)

    raise ValueError(f'Unsupported dataset hub: {data_source}')


def download_dataset_snapshot(
    data_id_or_path: str,
    data_source: Optional[str] = HubType.MODELSCOPE,
    revision: Optional[str] = None,
    force_redownload: bool = False,
    cache_dir: Optional[str] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
) -> str:
    """Download or resolve a dataset snapshot root from a supported hub."""
    data_source = _resolve_data_source(data_id_or_path, data_source)

    if data_source == HubType.LOCAL:
        root_dir = os.path.realpath(data_id_or_path)
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f'Local dataset directory was not found: {data_id_or_path}')
        return root_dir

    if data_source == HubType.HUGGINGFACE:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=data_id_or_path,
            repo_type='dataset',
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_redownload,
            allow_patterns=allow_file_pattern,
            ignore_patterns=ignore_file_pattern,
        )

    if data_source == HubType.MODELSCOPE:
        from modelscope import dataset_snapshot_download

        download_kwargs = {}
        if revision:
            download_kwargs['revision'] = revision
        if cache_dir:
            download_kwargs['cache_dir'] = cache_dir
        if allow_file_pattern is not None:
            download_kwargs['allow_file_pattern'] = allow_file_pattern
        if ignore_file_pattern is not None:
            download_kwargs['ignore_file_pattern'] = ignore_file_pattern
        return dataset_snapshot_download(data_id_or_path, **download_kwargs)

    raise ValueError(f'Unsupported dataset hub: {data_source}')
