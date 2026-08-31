#!/usr/bin/env python3
"""Generate the JSON Schema that is authoritative for the bundled Web API."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evalscope.service.api_models import WebApiContracts  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def _strip_property_titles(value: object) -> None:
    """Remove Pydantic's display titles, which codegen mistakes for named types."""
    if isinstance(value, dict):
        value.pop('title', None)
        for child in value.values():
            _strip_property_titles(child)
    elif isinstance(value, list):
        for child in value:
            _strip_property_titles(child)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    schema = WebApiContracts.model_json_schema(mode='serialization', by_alias=True)
    _strip_property_titles(schema)
    schema['title'] = 'WebApiContracts'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schema, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
