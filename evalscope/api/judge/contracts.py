"""The judge's output contract: a single JSON object, validated against a Pydantic schema.

The contract owns both halves of the format agreement -- the requirement written into the prompt
and the parser that reads the reply -- so the two cannot drift apart. Parsing is strict: a reply
that does not satisfy the schema is a ``parse_error``, never a silently-zero score.
"""
import json
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, List, Literal, Optional, Sequence, Type, get_args, get_origin


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

    parse_retries: int = Field(default=3, ge=0)
    """Extra attempts at the same case when the reply does not satisfy the schema. Benchmarks whose
    upstream defines a fallback verdict instead of a retry declare ``0``."""

    def instruction(self) -> str:
        """The format requirement appended to the benchmark's own prompt."""
        # The alias is the key the judge must actually emit when a schema declares one.
        fields = '\n'.join(
            f'- "{field.alias or name}": {_describe(field)}' for name, field in self.schema_model.model_fields.items()
        )
        return ('\n\nReply with a single JSON object and no other text, containing exactly these keys:\n'
                f'{fields}')

    def parse(self, response: str) -> ParseResult:
        """Parse a judge reply, strictly."""
        if not response:
            return ParseResult.failure('empty response')

        payloads = _payloads(response)
        if not payloads:
            return ParseResult.failure('no JSON object found in the reply')
        if len(payloads) > 1:
            return ParseResult.failure(f'{len(payloads)} JSON objects found where exactly one is required')

        try:
            data = json.loads(payloads[0])
        except json.JSONDecodeError as exc:
            return ParseResult.failure(f'reply is not valid JSON: {exc}')
        if not isinstance(data, dict):
            return ParseResult.failure(f'reply is a {type(data).__name__}, not a JSON object')

        try:
            return ParseResult.success(self.schema_model.model_validate(data))
        except Exception as exc:
            return ParseResult.failure(f'reply does not satisfy {self.schema_model.__name__}: {exc}')


def _payloads(response: str) -> List[str]:
    """Return every top-level JSON object in the reply.

    Scanning rather than anchoring lets a reasoning judge wrap its answer in a ``<think>`` block
    or a fence; requiring exactly one object keeps a two-verdict reply a failure, not a coin flip.
    """
    found: List[str] = []
    depth = 0
    start = 0
    in_string = False
    escaped = False
    for index, char in enumerate(response):
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            if depth == 0:
                start = index
            depth += 1
        elif char == '}' and depth:
            depth -= 1
            if depth == 0:
                found.append(response[start:index + 1])
    return found


def _describe(field: Any) -> str:
    """Describe one schema field for the prompt, so the judge is told the exact allowed values."""
    annotation = field.annotation
    if get_origin(annotation) is Literal:
        allowed = ' or '.join(f'"{value}"' for value in get_args(annotation))
        return f'exactly one of {allowed}'

    bounds = [
        f'>= {meta.ge}'
        if getattr(meta, 'ge', None) is not None else f'<= {meta.le}' if getattr(meta, 'le', None) is not None else ''
        for meta in field.metadata
    ]
    bounds = [bound for bound in bounds if bound]
    kind = (
        'true or false' if annotation is bool else 'an integer' if annotation is int else
        'a number' if annotation is float else 'a list' if get_origin(annotation) is list else 'a string'
    )
    if bounds:
        return f'{kind} {" and ".join(bounds)}'
    return f'{kind}' + (f' ({field.description})' if field.description else '')


__all__: Sequence[str] = ('OutputContract', 'ParseResult')
