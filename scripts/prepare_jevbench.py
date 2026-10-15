"""Build and optionally publish the lossless Benchmark Heaven public dataset mirror."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

COHORTS = {'original': 72, 'easy': 48, 'hard': 111}
SOURCE = 'https://github.com/fstandhartinger/jevbench'


def build_dataset(output: Path, revision: str, source_dir: Path | None = None) -> dict:
    """Preserve upstream bytes and make heterogeneous records Arrow-compatible."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Specify a full upstream Git commit SHA.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'source': SOURCE, 'revision': revision, 'files': {}, 'cohorts': COHORTS}
    kinds = Counter()
    ids = set()
    for cohort, expected_count in COHORTS.items():
        relative = f'datasets/public/{cohort}.jsonl'
        raw = (
            (source_dir / relative).read_bytes()
            if source_dir
            else urlopen(
                f'https://raw.githubusercontent.com/fstandhartinger/jevbench/{revision}/{relative}', timeout=60
            ).read()
        )
        records = [json.loads(line) for line in raw.decode('utf-8').split('\n') if line.strip()]
        if len(records) != expected_count:
            raise ValueError(f'{cohort}: expected {expected_count} records, got {len(records)}.')
        rows = []
        for record in records:
            if record['id'] in ids:
                raise ValueError(f'Duplicate source id: {record["id"]}')
            ids.add(record['id'])
            kinds[record['question']['type']] += 1
            if record['provenance']['license'] != 'MIT':
                raise ValueError(f'Unexpected source license: {record["id"]}')
            encoded = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
            assert json.loads(encoded) == record
            rows.append(
                {
                    'id': record['id'],
                    'cohort': cohort,
                    'family': record['family'],
                    'kind': record['question']['type'],
                    'record_json': encoded,
                }
            )
        original = output / 'original' / f'{cohort}.jsonl'
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(raw)
        path = output / 'data' / f'{cohort}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        restored = [json.loads(row['record_json']) for row in pq.read_table(path).to_pylist()]
        assert restored == records
        manifest['files'][relative] = hashlib.sha256(raw).hexdigest()
        manifest['files'][f'data/{cohort}.parquet'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for filename in ['LICENSE', 'THIRD-PARTY.md']:
        raw = (
            (source_dir / filename).read_bytes()
            if source_dir
            else urlopen(
                f'https://raw.githubusercontent.com/fstandhartinger/jevbench/{revision}/{filename}', timeout=60
            ).read()
        )
        (output / filename).write_bytes(raw)
        manifest['files'][filename] = hashlib.sha256(raw).hexdigest()
    manifest['kinds'] = dict(kinds)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    configs = [
        'configs:',
        '- config_name: default',
        '  default: true',
        '  data_files:',
        '  - split: test',
        '    path: data/*.parquet',
    ]
    for cohort in COHORTS:
        configs += [f'- config_name: {cohort}', '  data_files:', '  - split: test', f'    path: data/{cohort}.parquet']
    readme = '\n'.join(
        [
            '---',
            'license: mit',
            'language:',
            '- en',
            'task_categories:',
            '- text-classification',
            *configs,
            '---',
            '',
            '# JevBench public subset',
            '',
            f'Lossless mirror of [{SOURCE}]({SOURCE}) at commit `{revision}`.',
            '',
            "231 public decisions: original 72, easy 48, hard 111. This is Benchmark Heaven's",
            'typed-decision benchmark, not the separate JevBench metamorphic coherence project.',
            '',
            'Task counts: ' + ', '.join(f'{key}: {value}' for key, value in sorted(kinds.items())) + '.',
            '',
            '## Loading',
            '',
            '```python',
            'from modelscope.msdatasets import MsDataset',
            'import json',
            'ds = MsDataset.load("evalscope/jevbench", subset_name="default", split="test")',
            'record = json.loads(ds[0]["record_json"])',
            '```',
            '',
            'For Hugging Face datasets, first download this repository from ModelScope, then use',
            '`load_dataset("parquet", data_files={"test": "<local_snapshot>/data/*.parquet"}, split="test")`.',
            'This repository is hosted on ModelScope; it is not an HF dataset ID.',
            'Individual configs: `original`, `easy`, `hard`; all use the `test` split.',
            '',
            '## Schema and provenance',
            '',
            '`id`, `cohort`, `family`, `kind` are strings. `record_json` is the complete upstream record,',
            'including state, question, ordered labels, expected answer, group and provenance.',
            'JSON encoding avoids mixed string/object/numeric Arrow columns. No examples or labels are changed.',
            'Original source bytes are retained under `original/`; SHA-256 hashes are in `manifest.json`.',
            '',
            '## Evaluation limits',
            '',
            'Only the public 231 decisions are mirrored. Results are not the full JevBench leaderboard score.',
            'Do not train or calibrate on this evaluation set. Only state and question belong in model input;',
            'expected answers, rationales and provenance must never be sent to the model.',
            '',
            '## Attribution and license',
            '',
            'Benchmark Heaven / Florian Standhartinger. All mirrored records declare MIT in their provenance.',
            'Upstream LICENSE and THIRD-PARTY.md are included. No sealed or imported restricted tasks are mirrored.',
            '本镜像包含公开的 231 道决策题，保留原始数据及来源，不代表完整榜单测试集。',
            '',
        ]
    )
    (output / 'README.md').write_text(readme, encoding='utf-8')
    return manifest


def main() -> None:
    """Prepare a pinned snapshot and publish using SDK credentials outside the source tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    manifest = build_dataset(args.output, args.revision, args.source_dir)
    if args.publish:
        from modelscope.hub.api import HubApi

        api = HubApi()
        token = os.getenv('MODELSCOPE_API_TOKEN')
        try:
            api.get_dataset_id_and_type('jevbench', 'evalscope', token=token)
        except Exception as exc:
            # Do not turn authentication or transient failures into a repository creation.
            if '不存在的数据集' not in str(exc) and '404' not in str(exc):
                raise
            api.create_dataset(
                'jevbench',
                'evalscope',
                license='MIT',
                token=token,
                description='Benchmark Heaven JevBench public 231 decisions',
            )
        files = api.get_dataset_files('evalscope/jevbench', revision='master')
        entries = files.get('Files', []) if isinstance(files, dict) else files
        if any(
            not (
                entry.get('Path') == '.gitattributes'
                or (
                    entry.get('Path') == 'README.md'
                    and entry.get('CommitMessage', '').startswith('Commit .gitattributes README.md file(s)')
                )
            )
            for entry in entries
        ):
            raise ValueError('Refusing to overwrite a nonempty dataset repository.')
        api.upload_folder(
            repo_id='evalscope/jevbench',
            repo_type='dataset',
            folder_path=args.output,
            token=token,
            commit_message=f'Mirror public JevBench at {args.revision}',
        )
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
