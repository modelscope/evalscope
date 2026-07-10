from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator

from evalscope.api.dataset.hub import download_dataset_file, load_dataset_from_hub
from evalscope.constants import HubType
from evalscope.perf.domain.errors import PerfConfigError


@dataclass(frozen=True)
class DatasetResolver:
    """Resolve local and hub-backed datasets without embedding I/O policy in plugins."""

    data_source: str
    local_path: str | None = None

    def iter_lines(self, path: str | None = None) -> Iterator[str]:
        resolved = self._existing_file(path or self.local_path)
        with open(resolved, encoding='utf-8') as file:
            yield from file

    def iter_json_list(self, path: str | None = None) -> Iterator[Dict[str, Any]]:
        resolved = self._existing_file(path or self.local_path)
        try:
            with open(resolved, encoding='utf-8') as file:
                payload = json.load(file)
        except json.JSONDecodeError as e:
            raise PerfConfigError(f'Invalid JSON dataset {resolved}: {e}') from e
        if not isinstance(payload, list):
            raise PerfConfigError(f'JSON dataset {resolved} must contain a top-level list')
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise PerfConfigError(f'JSON dataset {resolved} item {index} must be an object')
            yield item

    def load(self, dataset_id: str, split: str = 'train', subset: str = 'default') -> Any:
        data_id_or_path = self.local_path or dataset_id
        source = HubType.LOCAL if self.local_path else self.data_source
        if self.local_path and not os.path.exists(self.local_path):
            raise PerfConfigError(f'Workload path {self.local_path!r} does not exist')
        return load_dataset_from_hub(
            data_id_or_path=data_id_or_path,
            split=split,
            subset=subset,
            data_source=source,
        )

    def resolve_file(self, dataset_id: str, file_name: str) -> str:
        if self.local_path and os.path.isfile(self.local_path):
            return self.local_path
        if self.local_path and os.path.isdir(self.local_path):
            candidate = os.path.join(self.local_path, file_name)
            if os.path.isfile(candidate):
                return candidate
            source = HubType.LOCAL
            data_id_or_path = self.local_path
        elif self.local_path:
            raise PerfConfigError(f'Workload path {self.local_path!r} does not exist')
        else:
            source = self.data_source
            data_id_or_path = dataset_id
        return download_dataset_file(
            data_id_or_path=data_id_or_path,
            file_path=file_name,
            data_source=source,
        )

    @staticmethod
    def _existing_file(path: str | None) -> str:
        if not path:
            raise PerfConfigError('A local dataset file is required')
        if not os.path.isfile(path):
            raise PerfConfigError(f'Local dataset file {path!r} does not exist')
        return path
