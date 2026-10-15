"""Lossless conversion of heterogeneous, Unicode-rich public decision records."""

import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

spec = importlib.util.spec_from_file_location('prepare_jevbench', Path(__file__).resolve().parents[2] / 'scripts/prepare_jevbench.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_231_records_and_configs_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    (source / 'datasets/public').mkdir(parents=True)
    originals = []
    for cohort, count in module.COHORTS.items():
        records = []
        for i in range(count):
            kind = ('choice', 'noul', 'score')[i % 3]
            record = {'id': f'{cohort}-{i}', 'family': 'unit', 'state': '中文\u2028line\u2029end' if i % 2 else {'nested': ['文本', i]},
                      'question': {'type': kind, 'instructions': 'Decide.',
                                   'criteria': {'B': '乙', 'A': '甲'} if kind == 'choice' else ['低', '中', '高'] if kind == 'score' else None},
                      'expected': 'B' if kind == 'choice' else 'yes' if kind == 'noul' else 2,
                      'provenance': {'license': 'MIT', 'source': 'unit'}}
            records.append(record)
        (source / f'datasets/public/{cohort}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        originals.extend(records)
    for name in ('LICENSE', 'THIRD-PARTY.md'):
        (source / name).write_text('MIT unit fixture\n')
    output = tmp_path / 'converted'
    manifest = module.build_dataset(output, 'a' * 40, source)
    restored = []
    for cohort, count in module.COHORTS.items():
        rows = pq.read_table(output / f'data/{cohort}.parquet').to_pylist()
        assert len(rows) == count
        restored.extend(json.loads(row['record_json']) for row in rows)
        assert (output / f'original/{cohort}.jsonl').read_bytes() == (source / f'datasets/public/{cohort}.jsonl').read_bytes()
    assert restored == originals
    assert len({r['id'] for r in restored}) == 231
    assert list(restored[0]['question']['criteria']) == ['B', 'A']
    card = yaml.safe_load((output / 'README.md').read_text().split('---')[1])
    assert [config['config_name'] for config in card['configs']] == ['default', 'original', 'easy', 'hard']
    assert sum(manifest['kinds'].values()) == 231
    assert all(len(value) == 64 for value in manifest['files'].values())


def test_reject_unpinned_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='full upstream Git commit'):
        module.build_dataset(tmp_path, 'main')
