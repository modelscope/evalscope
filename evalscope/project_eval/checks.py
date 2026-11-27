"""Read-only end-state checks for externally generated eval tasks."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests

from .schema import CheckSpec, EvalTask

_SENSITIVE_NAMES = {
    '.env',
    '.npmrc',
    '.pypirc',
    '.netrc',
}
_SENSITIVE_SUFFIXES = ('.pem', '.key', '.p12', '.pfx', '.kdbx')
_SENSITIVE_PREFIXES = ('id_rsa', 'id_ed25519', 'id_ecdsa', 'credentials')
_SENSITIVE_DIRS = {'.git', '.ssh', '.aws', '.gnupg', '.docker'}
_DEFAULT_HTTP_PREFIXES = ('http://127.0.0.1', 'http://localhost')


@dataclass
class CheckResult:
    check_id: str
    ok: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _is_sensitive(relative: Path) -> bool:
    for part in relative.parts:
        name = part.lower()
        if name in _SENSITIVE_DIRS:
            return True
        if name.startswith('secrets'):
            return True
    name = relative.name.lower()
    if name in _SENSITIVE_NAMES or name.startswith('.env'):
        return True
    if name.endswith(_SENSITIVE_SUFFIXES) or name.endswith('.env'):
        return True
    return name.startswith(_SENSITIVE_PREFIXES)


def assert_within_root(path: str, root: Path) -> Path:
    """Resolve a user path and reject escapes, symlinks and sensitive names."""
    if not path or path.startswith('/'):
        raise ValueError('path must be relative to the project root')
    if '..' in Path(path).parts:
        raise ValueError('path outside root')
    root_resolved = root.resolve()
    candidate = root_resolved / path
    for parent in (candidate, *candidate.parents):
        if parent == root_resolved:
            break
        if parent.is_symlink():
            raise ValueError('path must not be a symlink')
    resolved = candidate.resolve()
    if os.path.commonpath([str(root_resolved), str(resolved)]) != str(root_resolved):
        raise ValueError('path outside root')
    relative = resolved.relative_to(root_resolved)
    if _is_sensitive(relative):
        raise ValueError('sensitive path denied')
    return resolved


def _json_path_value(document: Any, json_path: str) -> Any:
    current = document
    for key in json_path.split('.'):
        if isinstance(current, list):
            current = current[int(key)]
        else:
            current = current[key]
    return current


def _run_check(check: CheckSpec, root: Path, allowed_http_prefixes: Iterable[str]) -> CheckResult:
    started = time.perf_counter()
    try:
        if check.type in {'file_exists', 'file_content', 'json_path'}:
            candidate = assert_within_root(check.path, root)
            if check.type == 'file_exists':
                return CheckResult(
                    check.id,
                    candidate.exists(),
                    {
                        'path': str(candidate.relative_to(root.resolve())),
                        'elapsed_ms': (time.perf_counter() - started) * 1000,
                    },
                )
            if check.type == 'file_content':
                text = candidate.read_text(encoding='utf-8', errors='replace')
                return CheckResult(
                    check.id,
                    check.contains in text,
                    {
                        'path': str(candidate.relative_to(root.resolve())),
                        'contains': check.contains,
                        'elapsed_ms': (time.perf_counter() - started) * 1000,
                    },
                )
            document = json.loads(candidate.read_text(encoding='utf-8'))
            value = _json_path_value(document, check.json_path)
            return CheckResult(
                check.id,
                value == check.expect,
                {
                    'path': str(candidate.relative_to(root.resolve())),
                    'json_path': check.json_path,
                    'value': value,
                    'expect': check.expect,
                    'elapsed_ms': (time.perf_counter() - started) * 1000,
                },
            )

        if check.type == 'sqlite':
            candidate = assert_within_root(check.dsn, root)
            connection = sqlite3.connect(f'file:{candidate}?mode=ro', uri=True)
            try:
                rows = connection.execute(check.query).fetchall()
            finally:
                connection.close()
            value = rows[0][0] if rows else None
            return CheckResult(
                check.id,
                str(value) == str(check.expect),
                {
                    'dsn': str(candidate.relative_to(root.resolve())),
                    'query': check.query,
                    'rows': [list(row) for row in rows],
                    'elapsed_ms': (time.perf_counter() - started) * 1000,
                },
            )

        if check.type == 'http':
            if not any(check.url.startswith(prefix) for prefix in allowed_http_prefixes):
                raise ValueError('http url is not allowed')
            response = requests.request(
                check.method,
                check.url,
                timeout=10,
                allow_redirects=False,
            )
            return CheckResult(
                check.id,
                response.status_code == check.expect_status,
                {
                    'url': check.url,
                    'status': response.status_code,
                    'expect_status': check.expect_status,
                    'elapsed_ms': (time.perf_counter() - started) * 1000,
                },
            )

        if check.type == 'http_json':
            if not any(check.url.startswith(prefix) for prefix in allowed_http_prefixes):
                raise ValueError('http url is not allowed')
            response = requests.get(check.url, timeout=10, allow_redirects=False)
            response.raise_for_status()
            value = _json_path_value(response.json(), check.json_path)
            return CheckResult(
                check.id,
                value == check.expect,
                {
                    'url': check.url,
                    'json_path': check.json_path,
                    'value': value,
                    'expect': check.expect,
                    'elapsed_ms': (time.perf_counter() - started) * 1000,
                },
            )

        raise ValueError(f'unsupported check type: {check.type}')
    except Exception as exc:  # noqa: BLE001 - reported as check evidence, never raised
        return CheckResult(
            check.id,
            False,
            {
                'elapsed_ms': (time.perf_counter() - started) * 1000,
            },
            error=f'{type(exc).__name__}: {exc}',
        )


def run_checks(
    task: EvalTask,
    root: Path,
    allowed_http_prefixes: Iterable[str] = _DEFAULT_HTTP_PREFIXES,
) -> list[CheckResult]:
    root_path = Path(root)
    return [_run_check(check, root_path, tuple(allowed_http_prefixes)) for check in task.checks]


__all__ = ['CheckResult', 'assert_within_root', 'run_checks']
