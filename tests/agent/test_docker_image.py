"""Tests for content-hashed Docker image helpers."""

from pathlib import Path

from evalscope.api.sandbox import hash_build_context


def test_hash_build_context_changes_for_source_and_cache_key(tmp_path: Path):
    source = tmp_path / 'runtime.py'
    source.write_text('VERSION = 1\n')
    initial_hash = hash_build_context(str(tmp_path), cache_key_parts=['runtime=v1'])

    source.write_text('VERSION = 2\n')

    assert hash_build_context(str(tmp_path), cache_key_parts=['runtime=v1']) != initial_hash
    assert hash_build_context(str(tmp_path), cache_key_parts=['runtime=v2']) != initial_hash
