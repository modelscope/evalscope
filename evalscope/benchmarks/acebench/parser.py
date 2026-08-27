# Copyright (c) Alibaba, Inc. and its affiliates.
"""Decoding of ACEBench model outputs.

Ports the official contract (``model_inference/utils.py::decode_ast`` plus the pre-processing
``eval_main.py`` applies before it): a prompt-mode answer must be a bracketed list of Python
calls, and anything else counts as ``wrong_output_format`` and scores zero.

Deviation from upstream: the official ``resolve_ast_by_type`` calls ``eval()`` on ``BinOp`` and
``Lambda`` argument nodes, which executes model-authored code. Here those nodes are resolved with
``ast.literal_eval`` and fall back to their source text, so no model output is ever executed.
No ACEBench sample carries such an argument, so scores are unaffected.
"""

import ast
from typing import Any, Dict, List

from .utils import extract_outermost_bracket_content

CallList = List[Dict[str, Dict[str, Any]]]


class CallFormatError(ValueError):
    """Raised when a model output does not follow ACEBench's call-list format."""


def decode_calls(raw_output: str, test_category: str) -> CallList:
    """Decode a prompt-mode model output into ``[{name: {arg: value}}]``.

    Args:
        raw_output: Raw text the model produced.
        test_category: Fine-grained ACEBench category, e.g. ``normal_atom_bool``.

    Returns:
        The decoded call list.

    Raises:
        CallFormatError: If the output is not a bracketed list of calls.
    """
    calls = _ast_parse(_preprocess(raw_output, test_category))
    if not _is_call_list(calls):
        raise CallFormatError('The output format does not meet the specified requirements.')
    return calls


def decode_execution_calls(message: str) -> CallList:
    """Decode an agent message so its calls can be executed during a rollout.

    Mirrors ``EXECUTION.decode_function_list``, which is more forgiving than the scoring decoder:
    brackets are added when missing and stripped before parsing, so a bare ``api(x=1)`` also runs.
    A nested argument call resolves to its source text here, matching the executor upstream.
    """
    text = message[1:] if message.startswith(' ') else message
    if not text.startswith('['):
        text = '[' + text
    if not text.endswith(']'):
        text = text + ']'

    body = ast.parse(text.strip("[]'"), mode='eval').body
    elements = body.elts if isinstance(body, (ast.Tuple, ast.List)) else [body]
    return [_resolve_call(element, nested_call_as_source=True) for element in elements if isinstance(element, ast.Call)]


def _preprocess(raw_output: Any, test_category: str) -> str:
    """Normalize the raw output exactly as the official evaluator does before decoding."""
    if not isinstance(raw_output, str):
        raise CallFormatError('Model output is not text.')

    if 'multi_turn' in test_category and 'agent' not in test_category:
        # normal_multi_turn_eval strips every whitespace character before decoding.
        return ''.join(raw_output.split())

    bracket_content = extract_outermost_bracket_content(raw_output)
    if bracket_content is None:
        raise CallFormatError('Model output does not contain a bracketed call list.')
    return bracket_content


def _ast_parse(text: str) -> CallList:
    """Parse ``[Api(key='value'), ...]`` into a call list, as ``ast_parse`` upstream does."""
    try:
        parsed = ast.parse(text, mode='eval')
    except (SyntaxError, ValueError) as exc:
        raise CallFormatError(f'Invalid syntax. Failed to decode AST. {exc}') from exc

    body = parsed.body
    if not isinstance(body, (ast.List, ast.Tuple)):
        # Upstream reads ``parsed.body.elts`` unconditionally and lets the AttributeError surface.
        raise CallFormatError('Model output is not a list of API calls.')

    calls = []
    for element in body.elts:
        if not isinstance(element, ast.Call):
            raise CallFormatError('Model output contains a list element that is not an API call.')
        calls.append(_resolve_call(element))
    return calls


def _resolve_call(node: ast.Call, nested_call_as_source: bool = False) -> Dict[str, Dict[str, Any]]:
    """Resolve a call node into ``{name: {arg: value}}``."""
    arguments = {
        keyword.arg: _resolve_value(keyword.value, nested_call_as_source)
        for keyword in node.keywords
        if keyword.arg is not None
    }
    return {_resolve_name(node.func): arguments}


def _resolve_name(node: ast.AST) -> str:
    """Flatten a possibly dotted callable reference into its source name."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return '.'.join(reversed(parts))


def _resolve_value(node: ast.AST, nested_call_as_source: bool = False) -> Any:
    """Resolve an argument node into a Python value, mirroring ``resolve_ast_by_type``.

    ``nested_call_as_source`` selects between the two upstream variants for an argument that is
    itself a call without keywords: the scorer keeps ``{name: {}}``, the executor keeps its source.
    """
    if isinstance(node, ast.Constant):
        return '...' if node.value is Ellipsis else node.value
    if isinstance(node, ast.UnaryOp):
        operand = _resolve_value(node.operand, nested_call_as_source)
        return -operand if isinstance(operand, (int, float)) and isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.List):
        return [_resolve_value(item, nested_call_as_source) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_resolve_value(item, nested_call_as_source) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _resolve_value(key, nested_call_as_source): _resolve_value(value, nested_call_as_source)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        if node.keywords:
            return _resolve_call(node, nested_call_as_source)
        return ast.unparse(node) if nested_call_as_source else {_resolve_name(node.func): {}}
    # Nodes upstream would ``eval()``; resolved without executing model-authored code.
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return ast.unparse(node)


def _is_call_list(decoded_output: Any) -> bool:
    """Port of ``is_function_call_format_valid``: the output must be a list of dicts."""
    return isinstance(decoded_output, list) and all(isinstance(item, dict) for item in decoded_output)
