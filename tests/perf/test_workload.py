import json
import pytest

from evalscope.perf.domain.errors import PerfConfigError
from evalscope.perf.workloads import DatasetResolver


def test_dataset_resolver_reads_json_and_jsonl_lazily(tmp_path) -> None:
    json_file = tmp_path / 'records.json'
    json_file.write_text(json.dumps([{'prompt': 'one'}, {'prompt': 'two'}]), encoding='utf-8')
    resolver = DatasetResolver(data_source='local', local_path=str(json_file))
    iterator = resolver.iter_json_list()
    assert next(iterator) == {'prompt': 'one'}

    jsonl_file = tmp_path / 'records.jsonl'
    jsonl_file.write_text('{"prompt":"one"}\n{"prompt":"two"}\n', encoding='utf-8')
    lines = DatasetResolver(data_source='local', local_path=str(jsonl_file)).iter_lines()
    assert next(lines).strip() == '{"prompt":"one"}'


def test_dataset_resolver_rejects_invalid_json_schema(tmp_path) -> None:
    path = tmp_path / 'invalid.json'
    path.write_text('{"not":"a list"}', encoding='utf-8')
    with pytest.raises(PerfConfigError, match='top-level list'):
        list(DatasetResolver(data_source='local', local_path=str(path)).iter_json_list())


def test_dataset_resolver_resolves_file_from_local_directory(tmp_path) -> None:
    path = tmp_path / 'dataset'
    path.mkdir()
    expected = path / 'records.jsonl'
    expected.write_text('{}\n', encoding='utf-8')
    resolver = DatasetResolver(data_source='local', local_path=str(path))
    assert resolver.resolve_file('unused', 'records.jsonl') == str(expected)
