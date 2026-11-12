"""The judge's output contract: a single JSON object, validated against a Pydantic schema.

The contract owns both halves of the format agreement -- the requirement written into the prompt
and the parser that reads the reply -- so the two cannot drift apart. A reply that does not
satisfy the schema is a ``parse_error``, never a silently-zero score.
"""

import json
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Type, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field


class ParseResult(BaseModel):
    """Outcome of parsing one judge response against a contract."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    value: Any = Field(default=None)
    error: Optional[str] = Field(default=None)

    @classmethod
    def success(cls, value: Any) -> 'ParseResult':
        return cls(ok=True, value=value)

    @classmethod
    def failure(cls, error: str) -> 'ParseResult':
        return cls(ok=False, error=error)


class OutputContract(BaseModel):
    """The JSON reply one judge case requires."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    schema_model: Type[BaseModel]

    def instruction(self) -> str:
        """The format requirement appended to the benchmark's own prompt."""
        # The alias is the key the judge must actually emit when a schema declares one.
        fields = '\n'.join(
            f'- "{field.alias or name}": {_describe(field)}' for name, field in self.schema_model.model_fields.items()
        )
        return (
            '\n\nReply with a single JSON object and no other text, containing these required keys '
            '(additional keys are allowed):\n'
            f'{fields}'
        )

    def parse(self, response: str) -> ParseResult:
        """Validate the entire final reply as one JSON object, then apply the Pydantic schema."""
        if not response or not response.strip():
            return ParseResult.failure('empty response')

        try:
            data = json.loads(response, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            return ParseResult.failure(f'reply is not valid JSON: {exc}')
        if not isinstance(data, dict):
            return ParseResult.failure(f'reply is a {type(data).__name__}, not a JSON object')

        try:
            return ParseResult.success(self.schema_model.model_validate(data))
        except Exception as exc:
            return ParseResult.failure(f'reply does not satisfy {self.schema_model.__name__}: {exc}')


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """Reject ambiguous duplicate keys instead of silently taking the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key!r}')
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f'invalid JSON constant: {value}')


def _describe(field: Any) -> str:
    """Describe one schema field for the prompt, so the judge is told the exact allowed values."""
    annotation = field.annotation
    nullable = type(None) in get_args(annotation)
    if nullable:
        variants = [item for item in get_args(annotation) if item is not type(None)]
        if len(variants) == 1:
            annotation = variants[0]
    if get_origin(annotation) is Literal:
        allowed = ' or '.join(f'"{value}"' for value in get_args(annotation))
        return f'exactly one of {allowed}'

    bounds = [
        f'>= {meta.ge}'
        if getattr(meta, 'ge', None) is not None
        else f'<= {meta.le}'
        if getattr(meta, 'le', None) is not None
        else ''
        for meta in field.metadata
    ]
    bounds = [bound for bound in bounds if bound]
    kind = (
        'true or false'
        if annotation is bool
        else 'an integer'
        if annotation is int
        else 'a number'
        if annotation is float
        else 'a list'
        if get_origin(annotation) is list
        else 'an object'
        if get_origin(annotation) is dict or isinstance(annotation, type) and issubclass(annotation, BaseModel)
        else 'a string'
    )
    if nullable:
        kind += ' or null'
    if bounds:
        return f'{kind} {" and ".join(bounds)}'
    return f'{kind}' + (f' ({field.description})' if field.description else '')


__all__: Sequence[str] = ('OutputContract', 'ParseResult')
