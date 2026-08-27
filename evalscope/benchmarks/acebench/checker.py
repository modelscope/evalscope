# Copyright (c) Alibaba, Inc. and its affiliates.
"""ACEBench answer checkers.

Faithful port of the official ``model_eval/checker.py`` and the category dispatch in
``model_eval/eval_main.py``. The control flow is kept deliberately close to upstream, including
behaviour that looks surprising, because ACEBench numbers are only comparable if the leniency
matches. The notable rules a reimplementation tends to get wrong:

* A parameter that appears in the ground truth but is not ``required`` by the schema may be
  omitted by the model and the call still counts as correct.
* When the ground-truth value's type differs from the declared schema type, the value check is
  skipped entirely (upstream's ``is_variable`` path).
* Strings are compared as substrings for ``normal``/``special`` data but for equality for
  ``agent`` data; lists are compared for strict equality; dicts compare key counts plus a
  substring test on the stringified value.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_STRING_NORMALIZE_RE = re.compile(r'[ \,\.\/\-\_\*\^]')
_TRAILING_INDEX_RE = re.compile(r'_\d+$')

PYTHON_TYPE_MAPPING = {
    'string': str,
    'integer': int,
    'float': float,
    'boolean': bool,
    'array': list,
    'tuple': list,
    'dict': dict,
    'any': str,
    'list': list,
    'object': dict,
    'objectArray': list,
    'list(string)': list,
    'list(enum)': list,
    'int': int,
    'enum': enumerate,
    'number': int,
}

PYTHON_NESTED_TYPE_CHECK_LIST = ['array', 'tuple', 'list(string)', 'list(enum)', 'object', 'objectArray']

CheckResult = Dict[str, Any]
CallList = List[Dict[str, Dict[str, Any]]]


def standardize_string(value: str) -> str:
    """Normalize a string the way ACEBench does before comparing it."""
    return _STRING_NORMALIZE_RE.sub('', value).lower().replace("'", '"')


def strip_answer_index(name: str) -> str:
    """Drop the ``_N`` suffix ACEBench uses to store repeated calls in a dict."""
    return _TRAILING_INDEX_RE.sub('', name)


def _valid() -> CheckResult:
    return {'valid': True, 'error': [], 'error_type': ''}


def _invalid(error: Any, error_type: str) -> CheckResult:
    return {'valid': False, 'error': error if isinstance(error, list) else [error], 'error_type': error_type}


def _value_error(param: str, func_name: str, expected: Any, actual: Any) -> str:
    """Render a value mismatch for the review file."""
    return f'wrong value for parameter ({param}) of api ({func_name}): [expected: {expected}, real: {actual}]'


def _possible_answer_type(possible_answer: Any) -> Optional[type]:
    """Port of ``get_possible_answer_type``: an empty string marks an optional parameter."""
    if possible_answer != '':
        return type(possible_answer)
    return None


def _coerce_bool_literal(value: Any) -> Any:
    """ACEBench accepts the strings ``true``/``false`` for booleans."""
    if value == 'true':
        return True
    if value == 'false':
        return False
    return value


def _find_description(func_descriptions: Any, name: str) -> Optional[Dict[str, Any]]:
    """Port of ``find_description``: the schema name is matched as a substring of ``name``."""
    if isinstance(func_descriptions, list):
        for func_description in func_descriptions:
            if func_description['name'] in name:
                return func_description
        return None
    return func_descriptions


def type_checker(
    param: str,
    value: Any,
    possible_answer: Any,
    expected_type_converted: Any,
    nested_type_converted: Any,
    func_name: str,
) -> CheckResult:
    """Check a parameter's type, reporting whether the value check should be skipped."""
    result: CheckResult = {'valid': True, 'error': [], 'is_variable': False, 'error_type': 'type_error'}

    # A ground-truth value whose type disagrees with the schema marks the parameter as a
    # "variable": upstream then trusts the type and skips value comparison entirely.
    is_variable = False
    possible_answer_type = _possible_answer_type(possible_answer)
    if possible_answer_type is not None and possible_answer_type != expected_type_converted:
        is_variable = True

    value = _coerce_bool_literal(value)

    if type(value) == expected_type_converted:  # noqa: E721
        if nested_type_converted is None:
            result['is_variable'] = is_variable
            return result
        for possible_answer_item in possible_answer:
            flag = True
            if type(possible_answer_item) == list:  # noqa: E721
                for value_item in value:
                    nested_result = type_checker(
                        param,
                        value_item,
                        possible_answer_item,
                        nested_type_converted,
                        None,
                        func_name,
                    )
                    if not nested_result['valid']:
                        flag = False
                        break
            if flag:
                return {'valid': True, 'error': [], 'is_variable': is_variable}
        result['valid'] = False
        result['error'] = [f'wrong inner type for parameter ({param}) of api ({func_name})']
        result['error_type'] = 'type_error'

    possible_answer_type = _possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if type(value) == possible_answer_type or possible_answer == value:  # noqa: E721
            result['is_variable'] = True
            return result

    result['valid'] = False
    result['error'] = [
        f'wrong type for parameter ({param}) of api ({func_name}): '
        f'[expected: {expected_type_converted}, real: {type(value)}]'
    ]
    result['error_type'] = 'type_error'
    return result


def string_checker(
    param: str, model_output: str, possible_answer: str, func_name: str, test_category: str
) -> CheckResult:
    """Compare strings: exact match for agent data, substring match otherwise."""
    normalized_output = standardize_string(model_output)
    normalized_answer = standardize_string(possible_answer)

    mismatch = (
        normalized_output != normalized_answer
        if 'agent' in test_category
        else normalized_answer not in normalized_output
    )
    if mismatch:
        return _invalid(_value_error(param, func_name, possible_answer, model_output), 'value_error:string')
    return _valid()


def list_checker(param: str, model_output: list, possible_answer: list, func_name: str) -> CheckResult:
    """Compare lists for strict equality after normalizing their string elements."""
    normalized_output = [standardize_string(item) if isinstance(item, str) else item for item in model_output]
    normalized_answer = [standardize_string(item) if isinstance(item, str) else item for item in possible_answer]

    if normalized_output != normalized_answer:
        return _invalid(_value_error(param, func_name, possible_answer, model_output), 'value_error:list/tuple')
    return _valid()


def dict_checker(param: str, model_output: Any, possible_answer: Any, func_name: str) -> CheckResult:
    """Compare dicts by key count, then per key with a substring test on the stringified value."""
    value_error = _invalid(_value_error(param, func_name, possible_answer, model_output), 'value_error')
    if not isinstance(model_output, dict):
        return value_error

    if len(model_output.keys()) != len(possible_answer.keys()):
        return value_error

    for key, value in model_output.items():
        value = _coerce_bool_literal(value)
        if key not in possible_answer:
            return value_error

        expected_value = possible_answer[key]
        if isinstance(expected_value, dict):
            nested_result = dict_checker(param, value, expected_value, func_name)
            if not nested_result['valid']:
                return nested_result
            continue

        normalized_value = standardize_string(value) if isinstance(value, str) else value
        normalized_answer = standardize_string(expected_value) if isinstance(expected_value, str) else expected_value

        if str(normalized_answer) not in str(normalized_value):
            return value_error

    return {'valid': True, 'error': [], 'error_type': 'dict_checker:unclear'}


def list_dict_checker(param: str, model_output: list, possible_answers: list, func_name: str) -> CheckResult:
    """Compare a list of dicts element-wise."""
    if len(model_output) != len(possible_answers):
        return _invalid(_value_error(param, func_name, possible_answers, model_output), 'value_error:list_dict_count')

    for predicted_item, expected_item in zip(model_output, possible_answers):
        result = dict_checker(param, predicted_item, expected_item, func_name)
        if not result['valid']:
            return result
    return {'valid': True, 'error': [], 'error_type': 'list_dict_checker:unclear'}


def simple_function_checker(
    func_description: Dict[str, Any],
    model_output: Dict[str, Dict[str, Any]],
    possible_answers: Dict[str, Any],
    test_category: str,
) -> CheckResult:
    """Check a single call against the schema and the ground-truth arguments."""
    possible_answer = list(possible_answers.values())[0]
    model_parameters = list(model_output.values())[0]
    schema_parameters = func_description['parameters']

    # Calls that legitimately take no arguments, e.g. [ApiName()].
    if model_parameters == {} and schema_parameters == {}:
        return _valid()
    if model_parameters == {} or schema_parameters == {}:
        return _invalid([], 'wrong_param')

    if possible_answer == schema_parameters['properties']:
        return _valid()
    if possible_answer == {} or schema_parameters == {}:
        return _invalid([], 'wrong_param')

    func_name = func_description['name']
    if func_name not in model_output:
        return _invalid(
            [{'wrong_function': {'expected': func_name, 'real': list(model_output.keys())[0]}}],
            'wrong_function_name',
        )

    param_details = schema_parameters['properties']
    for param in schema_parameters['required']:
        if param not in model_parameters:
            return _invalid(f'lack required_params: {param}', 'lack_args')

    for param, value in model_parameters.items():
        # Note the asymmetry that keeps EvalScope comparable with upstream: a parameter present in
        # the ground truth but absent from the model output is never checked unless it is required.
        if param not in param_details or param not in possible_answer:
            return _invalid(f'addition params: {param}', 'addition_args')

        expected_type_description = param_details[param]['type']
        expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
        nested_type_converted = None
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_type_converted = PYTHON_TYPE_MAPPING[_nested_type(param_details[param])]

        if expected_type_description == 'tuple' and isinstance(value, tuple):
            value = list(value)
        if expected_type_description == 'float' and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)

        type_result = type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_converted,
            nested_type_converted,
            func_name,
        )
        if not type_result['valid']:
            return _invalid(type_result['error'], type_result['error_type'])
        if type_result['is_variable']:
            continue

        result = _check_value(
            param,
            value,
            possible_answer[param],
            func_name,
            test_category,
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        if result is not None and not result['valid']:
            return _invalid(result['error'], result['error_type'])

    return _valid()


def _nested_type(param_detail: Dict[str, Any]) -> str:
    """Resolve the inner type of a container parameter, as upstream does."""
    try:
        return param_detail['items']['type']
    except (KeyError, TypeError):
        declared = param_detail['type']
        if 'string' in declared:
            return 'string'
        if 'enum' in declared:
            return 'enum'
        return 'dict'


def _check_value(
    param: str,
    value: Any,
    possible_answer: Any,
    func_name: str,
    test_category: str,
    expected_type_description: str,
    expected_type_converted: Any,
    nested_type_converted: Any,
) -> Optional[CheckResult]:
    """Dispatch to the per-type value checker; ``None`` means no check applies."""
    if expected_type_converted == dict:  # noqa: E721
        return dict_checker(param, value, possible_answer, func_name)
    if expected_type_converted == list and nested_type_converted == dict:  # noqa: E721
        if expected_type_description == 'objectArray' and len(value) != len(possible_answer):
            return _invalid('Wrong number of parameters for dictionary.', 'value_error:dict_items')
        return list_dict_checker(param, value, possible_answer, func_name)
    if expected_type_converted == str:  # noqa: E721
        return string_checker(param, value, possible_answer, func_name, test_category)
    if expected_type_converted == list:  # noqa: E721
        return list_checker(param, value, possible_answer, func_name)
    return None


def normal_checker(
    func_descriptions: List[Dict[str, Any]],
    model_output: CallList,
    possible_answers: Dict[str, Any],
    test_category: str,
) -> CheckResult:
    """Check a decoded call list against one ground-truth answer."""
    if len(model_output) != len(possible_answers):
        return _invalid(['The number of functions does not match the answer.'], 'wrong functions number')

    func_name_list = list(possible_answers.keys())
    # Ground truth stores repeated calls as ``name``/``name_1``; the suffix is not part of the API.
    answer_list = [{strip_answer_index(name): args} for name, args in possible_answers.items()]

    predicted_counts = _count_names(model_output)
    expected_counts = _count_names(answer_list)
    for name in predicted_counts:
        if name not in expected_counts:
            return _invalid([f'extra function detected: {name} is not in the ground truth'], 'function_mismatch')
    for name in expected_counts:
        if name not in predicted_counts:
            return _invalid([f'extra function detected: {name} is not in the ground truth'], 'function_mismatch')
    for name, count in predicted_counts.items():
        if count != expected_counts[name]:
            return _invalid(
                [f'incorrect count for function {name}: [expected: {expected_counts[name]}, actual: {count}]'],
                'function_mismatch',
            )

    result = _valid()
    for index, expected_call in enumerate(answer_list):
        expected_name = list(expected_call.keys())[0]
        func_description = _find_description(func_descriptions, func_name_list[index])
        valid_found = False
        for predicted_call in model_output:
            if list(predicted_call.keys())[0] == expected_name:
                result = simple_function_checker(func_description, predicted_call, expected_call, test_category)
                if result['valid']:
                    valid_found = True
                    break
            else:
                result = _invalid(['wrong_function'], 'simple_function_checker:unclear')
        if valid_found:
            continue
        if not result['valid']:
            if len(answer_list) > 1:
                result['error'] = [f'Parallel function call failed; excepted {answer_list}, real: {model_output}']
            return result
    return result


def _count_names(calls: CallList) -> Dict[str, int]:
    counter: Counter = Counter()
    for call in calls:
        counter.update(call.keys())
    return dict(counter)


def check_normal_answer(
    func_descriptions: List[Dict[str, Any]],
    model_output: CallList,
    possible_answer: Any,
    test_category: str,
) -> CheckResult:
    """Check a call list against every candidate ground-truth answer."""
    candidates = possible_answer if isinstance(possible_answer, list) else [possible_answer]
    errors = []
    for candidate in candidates:
        result = normal_checker(func_descriptions, model_output, candidate, test_category)
        if result['valid']:
            return result
        errors.append(result)
    return errors[0] if errors else _invalid(['No ground truth answer.'], 'missing_answer')


def check_special_answer(prediction: str, possible_answer: Any, test_category: str) -> CheckResult:
    """Check a special sample against ACEBench's diagnostic-string contract.

    The expected strings are English in both the ``en`` and ``zh`` datasets because the official
    Chinese prompt also asks for the English wording.
    """
    items = possible_answer.items() if isinstance(possible_answer, dict) else [('', [possible_answer])]
    result = _valid()

    if 'incomplete' in test_category:
        for name, values in items:
            missing = f'missing parameters ({values}) of ({name}) not pointed out'
            if 'Missing necessary parameters' not in prediction:
                result = _invalid(missing, 'error_detection')
            elif str(name) not in prediction or any(str(value) not in prediction for value in values):
                result = _invalid(missing, 'error_correction')
    elif 'error' in test_category:
        for name, values in items:
            wrong = f'incorrect values ({values}) of ({name}) not pointed out'
            if 'There is incorrect value' not in prediction:
                result = _invalid(wrong, 'error_detection')
            elif any(str(value) not in prediction for value in values):
                result = _invalid(wrong, 'error_correction')
    elif 'irrelevant' in test_category:
        if 'the limitations of the function' not in prediction:
            result = _invalid('request outside the function scope not pointed out', 'error_detection')
    else:
        result = _invalid(f'Unknown special ACEBench category: {test_category}', 'unknown_category')

    return result


def agent_state_checker(model_state: Dict[str, Any], possible_answer: Dict[str, Any]) -> CheckResult:
    """Compare one recorded API-class state against the expected end state."""
    result: CheckResult = {'valid': True, 'error': [], 'error_type': 'class attributes wrong'}

    scenario_name = list(possible_answer.keys())[0]
    expected_attributes = list(possible_answer.values())[0]
    model_attributes = list(model_state.values())[0]

    for attribute, model_value in model_attributes.items():
        if attribute not in expected_attributes:
            result['valid'] = False
            result['error'].append(f'class({scenario_name}) attributes({attribute}) missing in possible_answer.')
            continue

        expected_value = expected_attributes[attribute]
        if isinstance(expected_value, dict):
            for key, value in expected_value.items():
                if key not in model_value or value != model_value[key]:
                    result['valid'] = False
                    result['error'].append(
                        f'class({scenario_name}) attribute({attribute}.{key}) wrong, '
                        f'[expected: {value}, real: {model_value.get(key)}]'
                    )
        elif expected_value != model_value:
            result['valid'] = False
            result['error'].append(
                f'class({scenario_name}) attribute({attribute}) wrong, '
                f'[expected: {expected_value}, real: {model_value}]'
            )

    return result


def check_agent_end_state(model_states: List[Dict[str, Any]], possible_answer: Any) -> CheckResult:
    """Check the recorded end state of every involved API class.

    Mirrors ``agent_eval``: the number of classes must match, classes are paired by their key set,
    and a class whose key set finds no counterpart keeps the previous verdict (upstream behaviour,
    reproduced so that scores stay comparable).
    """
    expected_states = possible_answer if isinstance(possible_answer, list) else [possible_answer]
    if len(expected_states) != len(model_states):
        return _invalid([], 'wrong number of class')

    result = _valid()
    errors: List[Any] = []
    valid = True
    for expected_state in expected_states:
        expected_keys = set(expected_state.keys())
        matched = next((state for state in model_states if set(state.keys()) == expected_keys), None)
        if matched is not None:
            result = agent_state_checker(matched, expected_state)
        if not result['valid']:
            valid = False
            errors.append(result['error'])
    return _valid() if valid else {'valid': False, 'error': errors, 'error_type': 'class attributes wrong'}


def milestone_accuracy(process_trace: List[str], milestones: Any) -> float:
    """Compute ACEBench milestone (process) accuracy.

    Mirrors ``agent_eval_process``: each milestone is compared as a raw string against the recorded
    call trace and the denominator is always the declared milestone count, so a milestone that
    cannot be matched lowers the score instead of disappearing from the denominator.
    """
    candidates = milestones if _is_milestone_candidates(milestones) else [milestones]
    return max(_match_milestone_candidate(process_trace, candidate) for candidate in candidates)


def _is_milestone_candidates(milestones: Any) -> bool:
    """Return whether the milestones hold several alternative call sequences."""
    return (
        isinstance(milestones, list)
        and bool(milestones)
        and all(isinstance(candidate, list) for candidate in milestones)
    )


def _match_milestone_candidate(process_trace: List[str], candidate: Any) -> float:
    """Score one milestone candidate as an ordered subsequence of the process trace."""
    stones = candidate if isinstance(candidate, list) else [candidate]
    if not stones:
        # No milestone declared: vacuously satisfied, as in the official runner.
        return 1.0

    matched = 0
    cursor = 0
    for stone in stones:
        expected = str(stone).strip()
        while cursor < len(process_trace):
            if str(process_trace[cursor]).strip() == expected:
                matched += 1
                cursor += 1
                break
            cursor += 1
    return round(matched / len(stones), 3)


def multi_turn_accuracy(step_results: List[bool]) -> Tuple[float, float]:
    """Aggregate a ``normal_multi_turn`` dialogue: (end accuracy, process accuracy).

    Port of ``multiplt_turn_accuracy``: a dialogue only counts as correct when every step is
    correct, while the process score is the fraction of correct steps.
    """
    if not step_results:
        return 0.0, 0.0
    end_score = 0.0 if False in step_results else 1.0
    process_score = round(step_results.count(True) / len(step_results), 3)
    return end_score, process_score
