# Copyright (c) Alibaba, Inc. and its affiliates.
"""Benchmark plugin discovery.

Adapter modules are still discovered by globbing ``*/**/*_adapter.py``, but discovery
is separated from loading: ``_index.json`` maps every registered benchmark name to the
module that registers it, so resolving one benchmark imports one module instead of all
of them.

``_index.json`` is generated, never hand-written: ``evalscope benchmark-info
--update-index`` writes it and ``make docs-update`` / ``make docs-pipeline`` run that
step. A name missing from the index still resolves, by falling back to loading every
adapter module, so a stale or absent index can only cost time, never correctness.
"""

import glob
import importlib
import json
import os
from typing import Dict, List

from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.utils import get_logger

logger = get_logger()

_BENCHMARK_DIR = os.path.dirname(__file__)
_INDEX_PATH = os.path.join(_BENCHMARK_DIR, '_index.json')
_ADAPTER_PATTERN = os.path.join(_BENCHMARK_DIR, '*', '**', '*_adapter.py')

_loaded_all = False
_loading_all = False


def adapter_modules() -> List[str]:
    """Dotted paths of every adapter module the glob contract discovers."""
    modules = []
    for file_path in sorted(glob.glob(_ADAPTER_PATTERN, recursive=True)):
        if os.path.basename(file_path).startswith('_'):
            continue
        relative_path = os.path.relpath(file_path, _BENCHMARK_DIR)
        modules.append(f'evalscope.benchmarks.{relative_path[:-3].replace(os.path.sep, ".")}')
    return modules


def _read_index() -> Dict[str, str]:
    """Read the generated name -> module index, tolerating a missing or bad file."""
    try:
        with open(_INDEX_PATH, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug('Benchmark index %s not found; every lookup will load all adapters.', _INDEX_PATH)
    except (OSError, ValueError) as e:
        logger.warning('Ignoring unreadable benchmark index %s: %s', _INDEX_PATH, e)
    return {}


_INDEX: Dict[str, str] = _read_index()


def load_benchmark(name: str) -> bool:
    """Import only the module that registers ``name``.

    Returns:
        True when the name is indexed and its module has been imported, False when the
        index does not know the name, so the caller falls back to :func:`load_all`.
    """
    module = _INDEX.get(name)
    if module is None:
        return False
    importlib.import_module(module)
    return True


def load_all() -> None:
    """Import every adapter module, matching the historical eager behaviour."""
    global _loaded_all, _loading_all
    if _loaded_all or _loading_all:
        return
    _loading_all = True
    try:
        for module in adapter_modules():
            importlib.import_module(module)
        _loaded_all = True
    finally:
        _loading_all = False


def build_index() -> Dict[str, str]:
    """Map every registered benchmark name to the module that registers it.

    Names are not derivable from module paths: one module may register several names
    and a name may differ from its module leaf, which is why the mapping is read from
    the registry rather than guessed.
    """
    load_all()
    return {
        name: meta.data_adapter.__module__
        for name, meta in sorted(BENCHMARK_REGISTRY.items())
        if meta.data_adapter is not None
    }


def write_index() -> Dict[str, str]:
    """Regenerate ``_index.json`` from the registry and return the written mapping."""
    index = build_index()
    with open(_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=1, ensure_ascii=False, sort_keys=True)
        f.write('\n')
    logger.info('Wrote %d benchmark index entries to %s', len(index), _INDEX_PATH)
    return index


BENCHMARK_REGISTRY.set_resolvers(load_benchmark, load_all)
