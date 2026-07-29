"""Tests for :func:`evalscope.agent.external.helpers.extract_patch`.

Uses :class:`LocalAgentEnvironment` against a real on-disk git repository
so the helper is exercised through the same ``env.exec`` path adapters
will use in production (subprocess invocation, ``cwd=`` plumbing, error
propagation).
"""

import asyncio
import pytest
import subprocess
from pathlib import Path

from evalscope.agent.environments.local import LocalAgentEnvironment
from evalscope.agent.external.helpers import extract_patch


def _git(args, cwd):
    subprocess.run(['git', *args], cwd=str(cwd), check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    """Create a minimal git repo with one tracked file committed."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(['init', '-b', 'main'], cwd=repo)
    _git(['config', 'user.email', 'test@example.com'], cwd=repo)
    _git(['config', 'user.name', 'Test'], cwd=repo)
    (repo / 'main.py').write_text('def add(a, b):\n    return a - b\n')
    _git(['add', 'main.py'], cwd=repo)
    _git(['commit', '-m', 'initial'], cwd=repo)


def test_extract_patch_returns_modifications_against_head(tmp_path):
    repo = tmp_path / 'repo'
    _init_repo(repo)
    # Simulate the agent's edit: fix the bug.
    (repo / 'main.py').write_text('def add(a, b):\n    return a + b\n')

    env = LocalAgentEnvironment()
    patch = asyncio.run(extract_patch(env, cwd=str(repo)))

    assert patch, 'expected a non-empty diff for a real modification'
    assert 'diff --git' in patch
    assert '-    return a - b' in patch
    assert '+    return a + b' in patch


def test_extract_patch_excludes_untracked_files(tmp_path):
    """Untracked files are intentionally excluded — SWE-bench's eval
    container does ``git reset --hard {base_commit}`` which doesn't
    delete them, so including them in the diff trips ``git apply`` with
    ``already exists in working directory``."""
    repo = tmp_path / 'repo'
    _init_repo(repo)
    (repo / 'new_module.py').write_text('VALUE = 42\n')

    env = LocalAgentEnvironment()
    patch = asyncio.run(extract_patch(env, cwd=str(repo)))

    assert patch == ''
    assert 'new_module.py' not in patch


def test_extract_patch_returns_empty_when_clean(tmp_path):
    repo = tmp_path / 'repo'
    _init_repo(repo)

    env = LocalAgentEnvironment()
    patch = asyncio.run(extract_patch(env, cwd=str(repo)))

    assert patch == ''


def test_extract_patch_handles_non_git_dir(tmp_path, caplog):
    """Outside a git repository the helper should not raise — adapters
    feed the empty string straight into ``match_score`` which will score
    it as a failed attempt."""
    not_a_repo = tmp_path / 'plain'
    not_a_repo.mkdir()
    (not_a_repo / 'foo.txt').write_text('hello')

    env = LocalAgentEnvironment()
    patch = asyncio.run(extract_patch(env, cwd=str(not_a_repo)))

    assert patch == ''
