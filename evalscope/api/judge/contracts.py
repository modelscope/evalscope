"""The judge's output contract: a single JSON object, validated against a Pydantic schema.

The contract owns both halves of the format agreement -- the requirement written into the prompt
and the parser that reads the reply -- so the two cannot drift apart. A reply that does not
satisfy the schema is a ``parse_error``, never a silently-zero score.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, get_args, get_origin

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
        """Parse one judge reply through the schema's normal Pydantic coercion rules."""
        if not response:
            return ParseResult.failure('empty response')

        payloads, error = _payloads(response)
        if len(payloads) > 1:
            return ParseResult.failure(f'{len(payloads)} JSON objects found where exactly one is required')
        if not payloads:
            if error is not None:
                return ParseResult.failure(f'reply is not valid JSON: {error}')
            return ParseResult.failure('no JSON object found in the reply')

        try:
            return ParseResult.success(self.schema_model.model_validate(payloads[0]))
        except Exception as exc:
            return ParseResult.failure(f'reply does not satisfy {self.schema_model.__name__}: {exc}')


def _payloads(response: str) -> Tuple[List[Dict[str, Any]], Optional[json.JSONDecodeError]]:
    """Return every JSON object in the reply that is not nested in another, and the first malformed one's error.

    Scanning rather than anchoring lets a reasoning judge wrap its answer in a ``<think>`` block,
    a fence or an explanation; requiring exactly one object keeps a two-verdict reply a failure,
    not a coin flip. Decoding from each ``{`` skips braces and quotes in the surrounding text
    (``\\frac{1}{2}``, ``5" wide``) that do not start an object.
    """
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    error: Optional[json.JSONDecodeError] = None
    index = response.find('{')
    while index != -1:
        try:
            value, end = decoder.raw_decode(response, index)
        except json.JSONDecodeError as exc:
            # An object cut off by the end of the reply is missing, not malformed.
            if exc.pos < len(response):
                error = error or exc
            index = response.find('{', index + 1)
            continue
        found.append(value)
        index = response.find('{', end)
    return found, error


def _describe(field: Any) -> str:
    """Describe one schema field for the prompt, so the judge is told the exact allowed values."""
    annotation = field.annotation
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
        else 'a string'
    )
    if bounds:
        return f'{kind} {" and ".join(bounds)}'
    return f'{kind}' + (f' ({field.description})' if field.description else '')


__all__: Sequence[str] = ('OutputContract', 'ParseResult')
