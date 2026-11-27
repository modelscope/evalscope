"""Read-only end-state checks for externally generated eval tasks."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schema import RUNTIME_CHECKS_KEY, CheckSpec, EvalTask

_SENSITIVE_NAMES = {
    '.env',
    '.npmrc',
    '.pypirc',
    '.netrc',
}
_SENSITIVE_SUFFIXES = ('.pem', '.key', '.p12', '.pfx', '.kdbx')
_SENSITIVE_PREFIXES = ('id_rsa', 'id_ed25519', 'id_ecdsa', 'credentials')
_SENSITIVE_DIRS = {'.git', '.ssh', '.aws', '.gnupg', '.docker'}
DEFAULT_HTTP_PREFIXES = ('http://127.0.0.1', 'http://localhost', 'http://[::1]')


class CheckResult(BaseModel):
    """One read-only check result with evidence or an error."""

    model_config = ConfigDict(extra='forbid')

    check_id: str
    ok: bool
    evidence: dict[str, Any] = Field(default_factory=dict)
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


def _url_allowed(url: str, prefixes: Iterable[str]) -> bool:
    """Match a URL against trusted origins, not a raw string prefix."""
    try:
        target = urlsplit(url)
    except ValueError:
        return False
    if target.scheme not in {'http', 'https'} or not target.hostname:
        return False
    try:
        target_port = target.port
    except ValueError:
        return False
    effective_target_port = target_port if target_port is not None else (443 if target.scheme == 'https' else 80)
    for prefix in prefixes:
        try:
            allowed = urlsplit(prefix)
        except ValueError:
            continue
        if allowed.scheme not in {'http', 'https'} or not allowed.hostname:
            continue
        if target.scheme != allowed.scheme or target.hostname.lower() != allowed.hostname.lower():
            continue
        try:
            allowed_port = allowed.port
        except ValueError:
            continue
        if allowed_port is not None and effective_target_port != allowed_port:
            continue
        allowed_path = allowed.path or '/'
        if allowed_path == '/':
            return True
        if target.path == allowed_path or target.path.startswith(allowed_path.rstrip('/') + '/'):
            return True
    return False


def saved_checks_from_metadata(metadata: dict[str, Any] | None) -> list[CheckResult] | None:
    """Load the checks saved with a prediction, or None when they are unusable."""
    records = metadata.get(RUNTIME_CHECKS_KEY) if isinstance(metadata, dict) else None
    if not isinstance(records, list):
        return None
    try:
        return [CheckResult.model_validate(record) for record in records]
    except ValidationError:
        return None


def _json_path_value(document: Any, json_path: str) -> Any:
    current = document
    for key in json_path.split('.'):
        if isinstance(current, list):
            current = current[int(key)]
        else:
            current = current[key]
    return current


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def handle_file_exists(check: CheckSpec, root: Path, _: tuple[str, ...], started: float) -> CheckResult:
    """Check whether a file exists inside the state root."""
    candidate = assert_within_root(check.path, root)
    return CheckResult(
        check_id=check.id,
        ok=candidate.exists(),
        evidence={'path': str(candidate.relative_to(root.resolve())), 'elapsed_ms': _elapsed_ms(started)},
    )


def handle_file_content(check: CheckSpec, root: Path, _: tuple[str, ...], started: float) -> CheckResult:
    """Check whether a file contains the expected text."""
    contains = check.contains
    if contains is None:
        raise ValueError('file_content check requires contains')
    candidate = assert_within_root(check.path, root)
    text = candidate.read_text(encoding='utf-8', errors='replace')
    return CheckResult(
        check_id=check.id,
        ok=contains in text,
        evidence={
            'path': str(candidate.relative_to(root.resolve())),
            'contains': contains,
            'elapsed_ms': _elapsed_ms(started),
        },
    )


def handle_json_path(check: CheckSpec, root: Path, _: tuple[str, ...], started: float) -> CheckResult:
    """Compare a dotted JSON path in a file with the expected value."""
    candidate = assert_within_root(check.path, root)
    document = json.loads(candidate.read_text(encoding='utf-8'))
    value = _json_path_value(document, check.json_path)
    return CheckResult(
        check_id=check.id,
        ok=value == check.expect,
        evidence={
            'path': str(candidate.relative_to(root.resolve())),
            'json_path': check.json_path,
            'value': value,
            'expect': check.expect,
            'elapsed_ms': _elapsed_ms(started),
        },
    )


def handle_sqlite(check: CheckSpec, root: Path, _: tuple[str, ...], started: float) -> CheckResult:
    """Run a read-only SQLite query and compare its first value."""
    candidate = assert_within_root(check.dsn, root)
    connection = sqlite3.connect(f'{candidate.as_uri()}?mode=ro', uri=True)
    try:
        rows = connection.execute(check.query).fetchall()
    finally:
        connection.close()
    value = rows[0][0] if rows else None
    return CheckResult(
        check_id=check.id,
        ok=str(value) == str(check.expect),
        evidence={
            'dsn': str(candidate.relative_to(root.resolve())),
            'query': check.query,
            'rows': [list(row) for row in rows],
            'elapsed_ms': _elapsed_ms(started),
        },
    )


def handle_http(check: CheckSpec, _: Path, prefixes: tuple[str, ...], started: float) -> CheckResult:
    """Send one read-only HTTP request and compare its status."""
    if not _url_allowed(check.url, prefixes):
        raise ValueError('http url is not allowed')
    response = requests.request(check.method, check.url, timeout=10, allow_redirects=False)
    return CheckResult(
        check_id=check.id,
        ok=response.status_code == check.expect_status,
        evidence={
            'url': check.url,
            'status': response.status_code,
            'expect_status': check.expect_status,
            'elapsed_ms': _elapsed_ms(started),
        },
    )


def handle_http_json(check: CheckSpec, _: Path, prefixes: tuple[str, ...], started: float) -> CheckResult:
    """Fetch a read-only JSON endpoint and compare a dotted path value."""
    if not _url_allowed(check.url, prefixes):
        raise ValueError('http url is not allowed')
    response = requests.get(check.url, timeout=10, allow_redirects=False)
    response.raise_for_status()
    value = _json_path_value(response.json(), check.json_path)
    return CheckResult(
        check_id=check.id,
        ok=value == check.expect,
        evidence={
            'url': check.url,
            'json_path': check.json_path,
            'value': value,
            'expect': check.expect,
            'elapsed_ms': _elapsed_ms(started),
        },
    )


_CHECK_HANDLERS: dict[str, Callable[[CheckSpec, Path, tuple[str, ...], float], CheckResult]] = {
    'file_exists': handle_file_exists,
    'file_content': handle_file_content,
    'json_path': handle_json_path,
    'sqlite': handle_sqlite,
    'http': handle_http,
    'http_json': handle_http_json,
}


def _run_check(check: CheckSpec, root: Path, allowed_http_prefixes: tuple[str, ...]) -> CheckResult:
    started = time.perf_counter()
    try:
        handler = _CHECK_HANDLERS.get(check.type)
        if handler is None:
            raise ValueError(f'unsupported check type: {check.type}')
        return handler(check, root, allowed_http_prefixes, started)
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            ok=False,
            evidence={'elapsed_ms': _elapsed_ms(started)},
            error=f'{type(exc).__name__}: {exc}',
        )


def run_checks(
    task: EvalTask,
    root: Path,
    allowed_http_prefixes: Iterable[str] = DEFAULT_HTTP_PREFIXES,
) -> list[CheckResult]:
    """Run every read-only check for one task against the runtime state root."""
    root_path = Path(root)
    prefixes = tuple(allowed_http_prefixes)
    return [_run_check(check, root_path, prefixes) for check in task.checks]


__all__ = [
    'DEFAULT_HTTP_PREFIXES',
    'CheckResult',
    'assert_within_root',
    'run_checks',
    'saved_checks_from_metadata',
]
