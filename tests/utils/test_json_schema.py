from copy import deepcopy
from typing import Dict

import pytest
from pydantic import BaseModel

from evalscope.api.tool.tool_info import parse_tool_info
from evalscope.utils.json_schema import JSONSchema


@pytest.mark.parametrize('keyword', ['default', 'enum'])
def test_schema_conversion_preserves_literal_values(keyword: str) -> None:
    literal = {'type': 'str', 'nested': [{'type': 'int'}]}
    value = [literal] if keyword == 'enum' else literal
    raw = {'type': 'dict', keyword: value}
    original = deepcopy(raw)

    schema = JSONSchema.model_validate(raw)

    assert schema.type == 'object'
    assert getattr(schema, keyword) == value
    assert raw == original


@pytest.mark.parametrize(
    'nested',
    [
        {'properties': {'payload': {'type': 'dict', 'default': {'type': 'str'}}}},
        {'items': {'type': 'dict', 'default': {'type': 'str'}}},
        {'additionalProperties': {'type': 'dict', 'default': {'type': 'str'}}},
        {'anyOf': [{'type': 'dict', 'default': {'type': 'str'}}]},
    ],
)
def test_schema_conversion_still_normalizes_nested_schema_types(nested: dict) -> None:
    schema = JSONSchema.model_validate(nested).model_dump(exclude_none=True)
    child = next(iter(schema.values()))
    if 'properties' in schema:
        child = child['payload']
    elif 'anyOf' in schema:
        child = child[0]

    assert child['type'] == 'object'
    assert child['default'] == {'type': 'str'}


class Payload(BaseModel):
    options: Dict[str, str] = {'type': 'str'}


def test_tool_schema_preserves_pydantic_field_defaults() -> None:
    def tool(payload: Payload) -> str:
        """Accept a payload."""
        return payload.options['type']

    info = parse_tool_info(tool)

    assert info.parameters.properties['payload'].properties['options'].default == {'type': 'str'}
