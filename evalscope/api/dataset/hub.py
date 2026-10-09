# Copyright (c) Alibaba, Inc. and its affiliates.

import inspect
import os
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Union

from evalscope.constants import HubType
from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class DatasetHub:
    """Small hub handle shared by dataset loaders and benchmark-specific media resolvers."""

    data_id_or_path: str
    data_source: Optional[str] = HubType.MODELSCOPE
    revision: Optional[str] = None
    trust_remote: bool = True
    force_redownload: bool = False
    cache_dir: Optional[str] = None

    def load(self, split: str, subset: str = 'default', **kwargs):
        return load_dataset_from_hub(
            data_id_or_path=self.data_id_or_path,
            split=split,
            subset=subset,
            data_source=self.data_source,
            version=self.revision,
            trust_remote=self.trust_remote,
            force_redownload=self.force_redownload,
            **kwargs,
        )

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


def _prepare_cached_dataset_kwargs(load_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve snapshot files for validation and a distinct Hugging Face cache identity."""
    import datasets

    # Include legacy load options so older datasets versions do not forward them to the builder.
    load_only_options = {
        'split',
        'streaming',
        'num_proc',
        'keep_in_memory',
        'save_infos',
        'verification_mode',
        'ignore_verifications',
        'task',
    }
    builder_kwargs = {key: value for key, value in load_kwargs.items() if key not in load_only_options}
    builder = datasets.load_dataset_builder(**builder_kwargs)
    prepared_kwargs = dict(load_kwargs)
    if not builder.config.data_files:
        return prepared_kwargs
    if load_kwargs.get('data_files') is None:
        _validate_cached_shards(builder.config.data_files)
    # Absolute files and their metadata distinguish snapshots in the HF cache key.
    prepared_kwargs['data_files'] = builder.config.data_files
    return prepared_kwargs


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
    **kwargs,
):
    """Load a dataset split from ModelScope, Hugging Face, or a local dataset path."""
    import datasets
    from datasets import DownloadMode as HFDownloadMode

    data_source = _resolve_data_source(data_id_or_path, data_source)
    hf_download_mode = None if not force_redownload else HFDownloadMode.FORCE_REDOWNLOAD
    cached_snapshot = None
    if data_source == HubType.MODELSCOPE and not force_redownload:
        cached_snapshot = _try_modelscope_cached_snapshot(data_id_or_path, revision=version)

    if data_source in [HubType.HUGGINGFACE, HubType.LOCAL] or cached_snapshot:
        local_path = cached_snapshot or data_id_or_path
        # Hugging Face datasets may fail on local mirrors that contain a stale dataset_infos.json.
        dataset_infos_path = os.path.join(local_path, 'dataset_infos.json')
        if cached_snapshot is None and os.path.exists(dataset_infos_path):
            logger.info(f'Removing dataset_infos.json file at {dataset_infos_path} to avoid datasets errors.')
            os.remove(dataset_infos_path)

        load_kwargs = dict(
            path=local_path,
            name=subset if subset != 'default' else None,
            split=split,
            revision=version,
            download_mode=hf_download_mode,
            **kwargs,
        )
        if 'trust_remote_code' in inspect.signature(datasets.load_dataset).parameters:
            load_kwargs['trust_remote_code'] = trust_remote
        try:
            if cached_snapshot:
                load_kwargs = _prepare_cached_dataset_kwargs(load_kwargs)
            return datasets.load_dataset(**load_kwargs)
        except (FileNotFoundError, ValueError) as exc:
            if not cached_snapshot or not _is_missing_cached_data(exc):
                raise
            logger.info(
                f'Cached ModelScope dataset {data_id_or_path} cannot provide subset {subset}, split {split}: '
                f'{exc}. Falling back to MsDataset.load.'
            )

    if data_source == HubType.MODELSCOPE:
        from modelscope import MsDataset
        from modelscope.utils.constant import DownloadMode as MSDownloadMode

        load_kwargs = dict(
            dataset_name=data_id_or_path,
            split=split,
            subset_name=subset if subset != 'default' else None,
            trust_remote_code=trust_remote,
            **kwargs,
        )
        if version:
            load_kwargs['version'] = version
        if force_redownload:
            load_kwargs['download_mode'] = MSDownloadMode.FORCE_REDOWNLOAD
        dataset = MsDataset.load(**load_kwargs)
        if not isinstance(dataset, datasets.Dataset):
            dataset = dataset.to_hf_dataset()
        return dataset

    raise ValueError(f'Unsupported dataset hub: {data_source}')


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
