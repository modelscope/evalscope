# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import json
import re
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolInfo

_ROLE_RE = re.compile(r'^(user|system|assistant|tool):\s?(.*)$', re.IGNORECASE)
_STRING_NORMALIZE_RE = re.compile(r'[ \,\.\/\-\_\*\^]')
_TRAILING_INDEX_RE = re.compile(r'_\d+$')


def decode_maybe_json(value: Any, default: Any) -> Any:
    """Decode JSON strings while leaving already decoded values untouched."""
    if value is None or value == '':
        return deepcopy(default)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return deepcopy(default)
    return value


def normalize_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ACEBench tool schemas to EvalScope ToolInfo-compatible JSON schema."""
    normalized = deepcopy(tool)
    parameters = normalized.get('parameters') or {}
    normalized['parameters'] = _normalize_json_schema(parameters)
    if normalized['parameters'].get('type') != 'object':
        normalized['parameters']['type'] = 'object'
    normalized['parameters'].setdefault('properties', {})
    normalized['parameters'].setdefault('required', [])
    return normalized


def build_tool_infos(functions: List[Dict[str, Any]]) -> List[ToolInfo]:
    """Build EvalScope ToolInfo objects from ACEBench function specs."""
    return [ToolInfo.model_validate(normalize_tool_schema(function)) for function in functions]


def split_acebench_messages(question: str) -> List[Dict[str, str]]:
    """Split ACEBench's role-prefixed conversation string into chat-message dictionaries."""
    if not question:
        return [{'role': 'user', 'content': ''}]

    messages: List[Dict[str, str]] = []
    for line in question.splitlines():
        match = _ROLE_RE.match(line)
        if match:
            raw_role, content = match.groups()
            role = raw_role.lower()
            if role == 'system':
                role = 'assistant'
            messages.append({'role': role, 'content': content})
            continue

        if messages:
            messages[-1]['content'] = f'{messages[-1]["content"]}\n{line}'
        elif line.strip():
            messages.append({'role': 'user', 'content': line})

    return messages or [{'role': 'user', 'content': question}]


def extract_tool_calls_from_output(model_output: Optional[ModelOutput]) -> List[Dict[str, Dict[str, Any]]]:
    """Convert native model tool calls to ACEBench's ``[{name: args}]`` representation."""
    if model_output is None or model_output.empty:
        return []
    calls = []
    for tool_call in model_output.message.tool_calls or []:
        calls.append({tool_call.function.name: tool_call.function.arguments})
    return calls


def parse_call_list(text: Any) -> List[Dict[str, Dict[str, Any]]]:
    """Parse ACEBench text-form function calls into ``[{name: args}]``.

    Supports native-looking JSON lists/dicts and Python-call syntax such as
    ``[search(query='x'), save_item(id=1)]``.
    """
    if not isinstance(text, str) or not text:
        return []

    decoded = _loads_json_like(text)
    calls = _calls_from_json_like(decoded)
    if calls is not None:
        return calls

    bracket_text = extract_outermost_bracket_content(text)
    if bracket_text and bracket_text != text:
        decoded = _loads_json_like(bracket_text)
        calls = _calls_from_json_like(decoded)
        if calls is not None:
            return calls
        text = bracket_text

    return _calls_from_python_expr(text)


def score_normal_call(
    predicted_calls: List[Dict[str, Dict[str, Any]]],
    expected_answer: Any,
    test_category: str,
) -> Dict[str, Any]:
    """Score normal ACEBench function-call samples."""
    expected_candidates = expected_answer if isinstance(expected_answer, list) else [expected_answer]
    errors = []
    for expected in expected_candidates:
        valid, error = _match_call_set(predicted_calls, expected, strict_string=False)
        if valid:
            return {'valid': True, 'acc': 1.0, 'error': '', 'error_type': ''}
        errors.append(error)
    first_error = errors[0] if errors else {'error': 'No ground truth answer.', 'error_type': 'missing_answer'}
    return {
        'valid': False,
        'acc': 0.0,
        'error': first_error.get('error', ''),
        'error_type': first_error.get('error_type', ''),
        'test_category': test_category,
    }


def score_special_call(prediction: str, expected_answer: Any, test_category: str) -> Dict[str, Any]:
    """Score ACEBench special cases using the official string-diagnostic contract."""
    valid = True
    error = ''
    error_type = ''

    if 'incomplete' in test_category:
        for name, values in _iter_expected_items(expected_answer):
            if 'Missing necessary parameters' not in prediction or name not in prediction:
                valid = False
                error = (
                    f'The instruction is missing necessary parameters ({values}) for ({name}), '
                    'but the model failed to point it out.'
                )
                error_type = 'error_detection'
                break
            missing = [value for value in values if value not in prediction]
            if missing:
                valid = False
                error = (
                    f'The instruction is missing necessary parameters ({missing[0]}) for ({name}), '
                    'but the model failed to point it out.'
                )
                error_type = 'error_correction'
                break
    elif 'error' in test_category:
        for name, values in _iter_expected_items(expected_answer):
            if 'There is incorrect value' not in prediction:
                valid = False
                error = (
                    f'The instruction contains incorrect values ({values}) of ({name}), '
                    'but the model failed to point it out.'
                )
                error_type = 'error_detection'
                break
            missing = [str(value) for value in values if str(value) not in prediction]
            if missing:
                valid = False
                error = (
                    f'The instruction contains incorrect values ({values}) of ({name}), '
                    'but the model failed to point it out.'
                )
                error_type = 'error_correction'
                break
    elif 'irrelevant' in test_category:
        if 'the limitations of the function' not in prediction:
            valid = False
            error = 'The model should state that the request is outside the provided function scope.'
            error_type = 'error_detection'
    else:
        valid = False
        error = f'Unknown special ACEBench category: {test_category}'
        error_type = 'unknown_category'

    return {'valid': valid, 'acc': 1.0 if valid else 0.0, 'error': error, 'error_type': error_type}


def score_agent_call(
    prediction: str,
    predicted_calls: List[Dict[str, Dict[str, Any]]],
    expected_answer: Any,
    milestones: Any,
) -> Dict[str, Any]:
    """Score ACEBench agent samples.

    ACEBench's official runner executes tools and scores final environment state. EvalScope's
    default adapter path is static, so this scorer uses end-state accuracy when the prediction
    includes a final state, and otherwise reports call-process accuracy against the milestones.
    """
    state_prediction = extract_state_prediction(prediction)
    if state_prediction is not None and not _looks_like_expected_agent_state(state_prediction, expected_answer):
        state_prediction = None
    process_calls = extract_process_prediction(prediction, predicted_calls)
    process_acc = score_milestones(process_calls, milestones)

    value: Dict[str, float] = {'process_acc': process_acc}
    metadata: Dict[str, Any] = {'process_acc': process_acc}

    if state_prediction is not None:
        end_state_acc = 1.0 if match_agent_state(state_prediction, expected_answer) else 0.0
        value['end_state_acc'] = end_state_acc
        value['acc'] = end_state_acc
        metadata['end_state_acc'] = end_state_acc
    else:
        value['acc'] = process_acc
        metadata['end_state_acc'] = None

    return {
        'valid': bool(value['acc']),
        'acc': value['acc'],
        'process_acc': value['process_acc'],
        'end_state_acc': value.get('end_state_acc'),
        'error': '' if value['acc'] else 'Agent process or final state did not match the reference.',
        'error_type': '' if value['acc'] else 'agent_mismatch',
        'metadata': metadata,
    }


def extract_state_prediction(prediction: str) -> Optional[Any]:
    """Extract an agent final-state object from JSON-like model output."""
    decoded = _loads_json_like(prediction)
    if decoded is None:
        return None

    if isinstance(decoded, dict):
        for key in ['final_state', 'state', 'ground_truth', 'result']:
            if key in decoded:
                return decoded[key]
        if _looks_like_agent_state([decoded]):
            return [decoded]

    if isinstance(decoded, list) and _looks_like_agent_state(decoded):
        return decoded
    return None


def extract_process_prediction(
    prediction: str,
    predicted_calls: List[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Dict[str, Any]]]:
    """Extract agent process calls from native tool calls, JSON, or text call syntax."""
    if predicted_calls:
        return predicted_calls

    decoded = _loads_json_like(prediction)
    if isinstance(decoded, dict):
        for key in ['process', 'calls', 'tool_calls', 'mile_stone', 'milestone']:
            if key not in decoded:
                continue
            calls = _calls_from_json_like(decoded[key])
            if calls is not None:
                return calls
            if isinstance(decoded[key], list):
                return _parse_milestone_strings(decoded[key])

    return parse_call_list(prediction)


def score_milestones(predicted_calls: List[Dict[str, Dict[str, Any]]], milestones: Any) -> float:
    """Compute subsequence match accuracy for ACEBench agent milestones."""
    if not milestones:
        return 1.0

    candidates = milestones if _is_milestone_candidates(milestones) else [milestones]
    best = 0.0
    for candidate in candidates:
        expected_calls = _parse_milestone_strings(candidate)
        if not expected_calls:
            best = max(best, 1.0)
            continue
        matched = _count_subsequence_matches(predicted_calls, expected_calls)
        best = max(best, round(matched / len(expected_calls), 3))
    return best


def match_agent_state(prediction: Any, expected_answer: Any) -> bool:
    """Match ACEBench agent final state, ignoring top-level class order."""
    expected_list = expected_answer if isinstance(expected_answer, list) else [expected_answer]
    prediction_list = prediction if isinstance(prediction, list) else [prediction]
    if len(expected_list) != len(prediction_list):
        return False

    unmatched = list(prediction_list)
    for expected_item in expected_list:
        expected_keys = set(expected_item.keys()) if isinstance(expected_item, dict) else set()
        match_index = next(
            (
                index for index, predicted_item in enumerate(unmatched)
                if isinstance(predicted_item, dict) and set(predicted_item.keys()) == expected_keys
            ),
            None,
        )
        if match_index is None:
            return False
        predicted_item = unmatched.pop(match_index)
        if not _values_match(expected_item, predicted_item, strict_string=True):
            return False
    return True


def extract_outermost_bracket_content(text: str) -> Optional[str]:
    """Return the first balanced ``[...]`` block from text."""
    start = -1
    depth = 0
    for index, char in enumerate(text):
        if char == '[':
            if depth == 0:
                start = index
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start != -1:
                return text[start:index + 1]
    return None


def _normalize_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key == 'type' and item == 'dict':
                normalized[key] = 'object'
            else:
                normalized[key] = _normalize_json_schema(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_json_schema(item) for item in value]
    return value


def _loads_json_like(text: Any) -> Optional[Any]:
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None


def _calls_from_json_like(value: Any) -> Optional[List[Dict[str, Dict[str, Any]]]]:
    if value is None:
        return None
    if isinstance(value, dict):
        if 'function' in value:
            return [_call_from_openai_tool(value)]
        if 'name' in value and 'arguments' in value:
            return [{str(value['name']): _coerce_arguments(value.get('arguments'))}]
        if 'tool_calls' in value:
            return _calls_from_json_like(value['tool_calls'])
        if value and all(isinstance(args, dict) for args in value.values()):
            return [{str(name): args} for name, args in value.items()]
        return None
    if isinstance(value, list):
        calls: List[Dict[str, Dict[str, Any]]] = []
        for item in value:
            item_calls = _calls_from_json_like(item)
            if item_calls is None:
                return None
            calls.extend(item_calls)
        return calls
    return None


def _call_from_openai_tool(value: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    function = value['function']
    return {str(function['name']): _coerce_arguments(function.get('arguments', {}))}


def _coerce_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        decoded = _loads_json_like(arguments)
        if isinstance(decoded, dict):
            return decoded
    return {}


def _calls_from_python_expr(text: str) -> List[Dict[str, Dict[str, Any]]]:
    try:
        parsed = ast.parse(text.strip(), mode='eval')
    except Exception:
        return []

    body = parsed.body
    elements = body.elts if isinstance(body, (ast.List, ast.Tuple)) else [body]
    calls = []
    for element in elements:
        if isinstance(element, ast.Call):
            calls.append(_call_from_ast(element))
        elif isinstance(element, ast.Dict):
            literal = _literal_from_ast(element)
            item_calls = _calls_from_json_like(literal)
            if item_calls is not None:
                calls.extend(item_calls)
    return calls


def _call_from_ast(call: ast.Call) -> Dict[str, Dict[str, Any]]:
    name = _function_name(call.func)
    arguments = {}
    for keyword in call.keywords:
        if keyword.arg is not None:
            arguments[keyword.arg] = _literal_from_ast(keyword.value)
    return {name: arguments}


def _function_name(node: ast.AST) -> str:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return '.'.join(reversed(parts))


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _call_from_ast(node)
    return ast.unparse(node)


def _match_call_set(
    predicted_calls: List[Dict[str, Dict[str, Any]]],
    expected_answer: Dict[str, Any],
    strict_string: bool,
) -> Tuple[bool, Dict[str, str]]:
    expected_calls = [{_strip_answer_index(name): args} for name, args in expected_answer.items()]
    predicted_names = [_strip_answer_index(next(iter(call.keys()))) for call in predicted_calls if call]
    expected_names = [_strip_answer_index(next(iter(call.keys()))) for call in expected_calls if call]
    if Counter(predicted_names) != Counter(expected_names):
        return False, {
            'error': f'Function set mismatch: expected {expected_names}, got {predicted_names}.',
            'error_type': 'function_mismatch',
        }

    unmatched = list(predicted_calls)
    for expected_call in expected_calls:
        expected_name, expected_args = next(iter(expected_call.items()))
        match_index = next(
            (
                index for index, predicted_call in enumerate(unmatched)
                if _call_matches(predicted_call, expected_name, expected_args, strict_string)
            ),
            None,
        )
        if match_index is None:
            return False, {
                'error': f'No matching arguments found for function {expected_name}.',
                'error_type': 'argument_mismatch',
            }
        unmatched.pop(match_index)

    return True, {'error': '', 'error_type': ''}


def _call_matches(
    predicted_call: Dict[str, Dict[str, Any]],
    expected_name: str,
    expected_args: Dict[str, Any],
    strict_string: bool,
) -> bool:
    if not predicted_call:
        return False
    predicted_name, predicted_args = next(iter(predicted_call.items()))
    if _strip_answer_index(predicted_name) != expected_name:
        return False
    if set(predicted_args.keys()) != set(expected_args.keys()):
        return False
    return _values_match(expected_args, predicted_args, strict_string=strict_string)


def _values_match(expected: Any, actual: Any, strict_string: bool) -> bool:
    actual = _coerce_bool_string(actual)
    expected = _coerce_bool_string(expected)

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual.keys()) != set(expected.keys()):
            return False
        return all(_values_match(expected[key], actual[key], strict_string=strict_string) for key in expected)

    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_values_match(exp, act, strict_string=strict_string) for exp, act in zip(expected, actual))

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)

    if isinstance(expected, str) and isinstance(actual, str):
        if strict_string:
            return _standardize_string(expected) == _standardize_string(actual)
        return _standardize_string(expected) in _standardize_string(actual)

    return expected == actual


def _coerce_bool_string(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == 'true':
            return True
        if normalized == 'false':
            return False
    return value


def _standardize_string(value: str) -> str:
    return _STRING_NORMALIZE_RE.sub('', value).lower().replace("'", '"')


def _strip_answer_index(name: str) -> str:
    return _TRAILING_INDEX_RE.sub('', name)


def _iter_expected_items(expected_answer: Any) -> List[Tuple[str, List[str]]]:
    if isinstance(expected_answer, dict):
        return [(str(name), [str(value) for value in values]) for name, values in expected_answer.items()]
    return [('', [str(expected_answer)])]


def _parse_milestone_strings(milestones: Any) -> List[Dict[str, Dict[str, Any]]]:
    calls: List[Dict[str, Dict[str, Any]]] = []
    if not isinstance(milestones, list):
        milestones = [milestones]
    for milestone in milestones:
        if isinstance(milestone, str):
            calls.extend(parse_call_list(milestone))
        else:
            item_calls = _calls_from_json_like(milestone)
            if item_calls is not None:
                calls.extend(item_calls)
    return calls


def _is_milestone_candidates(milestones: Any) -> bool:
    return (
        isinstance(milestones, list) and milestones and all(isinstance(candidate, list) for candidate in milestones)
    )


def _count_subsequence_matches(
    predicted_calls: List[Dict[str, Dict[str, Any]]],
    expected_calls: List[Dict[str, Dict[str, Any]]],
) -> int:
    current = 0
    matched = 0
    for expected in expected_calls:
        while current < len(predicted_calls):
            if _same_call(predicted_calls[current], expected):
                matched += 1
                current += 1
                break
            current += 1
    return matched


def _same_call(left: Dict[str, Dict[str, Any]], right: Dict[str, Dict[str, Any]]) -> bool:
    if not left or not right:
        return False
    left_name, left_args = next(iter(left.items()))
    right_name, right_args = next(iter(right.items()))
    return _strip_answer_index(left_name) == _strip_answer_index(right_name) and _values_match(
        right_args, left_args, strict_string=True
    )


def _looks_like_agent_state(value: Any) -> bool:
    return isinstance(value, list) and value and all(isinstance(item, dict) for item in value)


def _looks_like_expected_agent_state(prediction: Any, expected_answer: Any) -> bool:
    expected_list = expected_answer if isinstance(expected_answer, list) else [expected_answer]
    prediction_list = prediction if isinstance(prediction, list) else [prediction]
    expected_keys = _state_key_counter(expected_list)
    predicted_keys = _state_key_counter(prediction_list)
    return bool(expected_keys) and expected_keys == predicted_keys


def _state_key_counter(items: List[Any]) -> Counter:
    return Counter(tuple(sorted(item.keys())) for item in items if isinstance(item, dict))
