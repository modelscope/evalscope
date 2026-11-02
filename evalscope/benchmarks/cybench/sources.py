"""Pinned public handouts and service sources from the GlacierCTF authors."""

import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import List, Optional
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, field_validator

from evalscope.api.dataset import download_dataset_file
from evalscope.utils.download_utils import download_url, file_sha256

GLACIER_REPO = 'LosFuzzys/GlacierCTF2023_writeups'
GLACIER_SHA = 'af41b3be06c86d20585e4c11c4b294659a91a469'


class GlacierFile(BaseModel):
    """One source file and its optional public destination, relative to its task directory."""

    model_config = ConfigDict(extra='forbid')
    path: str
    sha256: str
    target: Optional[str] = None
    executable: bool = False

    @field_validator('path', 'target')
    @classmethod
    def relative_path(cls, value: Optional[str]) -> Optional[str]:
        """Keep source and destination paths inside the selected task."""
        if value is not None and (not value or PurePosixPath(value).is_absolute() or '..' in value.split('/')):
            raise ValueError('GlacierCTF file paths must be relative and cannot contain parent traversal.')
        return value


class GlacierTask(BaseModel):
    """Native launch data reconstructed from the original competition sources, not an Inspect protocol."""

    model_config = ConfigDict(extra='forbid')
    path: str
    revision: str = GLACIER_SHA
    description: str
    service: bool
    reference: GlacierFile
    files: List[GlacierFile]

    @field_validator('path')
    @classmethod
    def relative_path(cls, value: str) -> str:
        return GlacierFile.relative_path(value)


def stage_glacier_task(
    task: GlacierTask,
    cache_dir: str,
    source_root: Optional[str] = None,
) -> tempfile.TemporaryDirectory:
    """Verify an allowlist of author files and stage only those files for the sample runtime.

    GitHub is not a dataset hub; its pinned raw files use the shared atomic URL downloader.
    Local checkouts use the dataset file resolver's existing path-safety checks.
    """
    if task.revision != GLACIER_SHA:
        raise ValueError('GlacierCTF source revision differs from the pinned manifest.')
    staging = tempfile.TemporaryDirectory(prefix='evalscope-cybench-glacier-')
    try:
        cache = Path(cache_dir) / 'cybench' / GLACIER_SHA
        for file in task.files:
            relative = f'{task.path}/{file.path}'
            if source_root:
                source = Path(download_dataset_file(source_root, relative))
            else:
                source = cache / relative
                download_url(
                    f'https://raw.githubusercontent.com/{GLACIER_REPO}/{GLACIER_SHA}/{quote(relative)}',
                    str(source),
                    sha256=file.sha256,
                )
            if file_sha256(str(source)) != file.sha256:
                raise ValueError(f'GlacierCTF source checksum mismatch: {relative}')
            destination = Path(staging.name) / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o755 if file.executable else 0o644)
        return staging
    except BaseException:
        staging.cleanup()
        raise
