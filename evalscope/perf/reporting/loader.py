import json
import os

from evalscope.perf.domain.errors import ResultStoreError
from evalscope.perf.domain.result import PerfSuiteResult


def load_suite(path: str) -> PerfSuiteResult:
    """Load a suite result from its root directory or summary JSON path."""
    manifest_path = os.path.join(path, 'manifest.json') if os.path.isdir(path) else path
    try:
        if os.path.basename(manifest_path) == 'manifest.json':
            with open(manifest_path, encoding='utf-8') as file:
                manifest = json.load(file)
            summary_path = manifest['files']['summary']
        else:
            summary_path = path
        with open(summary_path, encoding='utf-8') as file:
            return PerfSuiteResult.model_validate(json.load(file))
    except (KeyError, OSError, ValueError) as e:
        raise ResultStoreError(f'Unable to load suite result from {path}: {e}') from e
