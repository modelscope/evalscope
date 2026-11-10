"""Regenerate the bundled E2B image mapping from a CSV image catalog."""

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

DEFAULT_DATASETS = (
    'SWE-bench Verified',
    'SWE-bench Full Test',
    'SWE-bench Multilingual',
    'SWE-bench Pro',
    'TB 2.1',
    'DeepSWE',
    'SWE-Atlas',
    'Toolathlon',
)
REQUIRED_COLUMNS = {'源镜像', '转换后镜像', '数据集'}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'evalscope/api/sandbox/e2b_image_mapping.json.gz'


def build_mapping(rows: Iterable[Mapping[str, str]], datasets: Sequence[str]) -> Dict[str, str]:
    """Select nonempty converted images and reject conflicting source mappings."""
    selected = set(datasets)
    mapping, conflicts = {}, set()
    for row in rows:
        if row['数据集'] not in selected or not row['转换后镜像'].strip():
            continue
        source, target = row['源镜像'].strip(), row['转换后镜像'].strip()
        if source in mapping and mapping[source] != target:
            conflicts.add(source)
        mapping[source] = target
    if conflicts:
        raise ValueError(f'Conflicting source images: {sorted(conflicts)[:10]}')
    return mapping


def write_mapping(path: Path, mapping: Mapping[str, str]) -> None:
    """Write byte-reproducible sorted UTF-8 JSON inside gzip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(sorted(mapping.items())), ensure_ascii=False, indent=2).encode('utf-8')
    with path.open('wb') as output:
        with gzip.GzipFile(fileobj=output, filename='', mode='wb', mtime=0) as compressed:
            compressed.write(payload)


def main() -> None:
    """Filter the catalog and write the bundled compressed mapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dataset', action='append')
    args = parser.parse_args()
    datasets = args.dataset or DEFAULT_DATASETS
    try:
        with args.csv.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f'Missing columns: {sorted(missing)}')
            rows = list(reader)
        mapping = build_mapping(rows, datasets)
        write_mapping(args.output, mapping)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{args.csv}: {exc}\n')
    counts = Counter(row['数据集'] for row in rows if row['数据集'] in datasets and row['转换后镜像'].strip())
    for dataset in datasets:
        print(f'{dataset}: {counts[dataset]}')
    print(f'Total: {len(mapping)}')


if __name__ == '__main__':
    main()
