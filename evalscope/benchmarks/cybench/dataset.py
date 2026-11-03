"""Cybench task records and verified task archive staging."""

import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import List

from pydantic import BaseModel, ConfigDict, field_validator

from evalscope.utils.download_utils import file_sha256


class CybenchTask(BaseModel):
    """One task record downloaded from the benchmark dataset."""

    model_config = ConfigDict(extra='forbid')
    path: str
    category: str
    difficulty: str
    target_host: str
    answer: str
    answer_format: str
    metadata_sha256: str
    archive: str
    archive_sha256: str
    statement: str = ''

    @field_validator('path', 'archive')
    @classmethod
    def relative_path(cls, value: str) -> str:
        """Keep task and archive paths relative to their runtime or dataset root."""
        if not value or PurePosixPath(value).is_absolute() or '..' in value.split('/'):
            raise ValueError('Cybench paths must be relative and cannot contain parent traversal.')
        return value


class CybenchManifest(BaseModel):
    """Dataset task catalog and the upstream scoring source revision."""

    model_config = ConfigDict(extra='forbid')
    source_sha: str
    tasks: List[CybenchTask]


def stage_task_archive(path: str, expected_sha256: str) -> tempfile.TemporaryDirectory:
    """Verify a downloaded task archive and extract regular files into sample-owned staging."""
    if file_sha256(path) != expected_sha256:
        raise ValueError('Cybench task archive checksum mismatch.')
    staging = tempfile.TemporaryDirectory(prefix='evalscope-cybench-task-')
    try:
        with tarfile.open(path, 'r:gz') as archive:
            for item in archive.getmembers():
                name = PurePosixPath(item.name)
                if name.is_absolute() or '..' in name.parts or not (item.isfile() or item.isdir()):
                    raise ValueError(f'Unsafe Cybench archive member: {item.name}')
            archive.extractall(Path(staging.name))
        return staging
    except BaseException:
        staging.cleanup()
        raise
