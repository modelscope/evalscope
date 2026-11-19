# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import re
from typing import Any, Dict


def format_grid(grid: list) -> str:
    """Format a 2D grid as a JSON array string."""
    return json.dumps(grid)


def build_task_prompt(record: Dict[str, Any], pair_index: int = 0) -> str:
    """Build the prompt for one test input of an ARC-AGI-2 task from a dataset record."""
    fewshots = record['fewshots']
    question = record['question']

    parts = []
    parts.append(
        'Each grid is a 2D array of integers (0-9). '
        'Study the pattern in the examples, then predict the output for the test input.'
    )
    parts.append('')
    parts.append('Examples:')

    for i, pair in enumerate(fewshots, 1):
        parts.append(f'Example {i}:')
        parts.append(f'Input: {format_grid(pair["input"])}')
        parts.append(f'Output: {format_grid(pair["output"])}')
        parts.append('')

    test_input = question[pair_index]['input']
    parts.append(f'Test Input: {format_grid(test_input)}')
    parts.append('')
    parts.append('Provide the output grid as a JSON 2D array. Only output the JSON array, nothing else.')

    return '\n'.join(parts)


def _decode_grid(text: str) -> list | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, list) and all(isinstance(row, list) for row in value):
        return value
    return None


def parse_grid_from_response(response: str) -> list:
    """Use the boxed answer first, then the final bracketed value, as ARC Prize does."""
    boxed = re.search(r'\\boxed\{(.*?)\}', response, re.DOTALL)
    if boxed:
        grid = _decode_grid(boxed[1])
        if grid is not None:
            return grid

    end = response.rfind(']')
    if end == -1 or response.rfind('}') > end:
        return []

    # Match the outer array ending here, including all rows of a multi-row grid.
    depth = 0
    for start in range(end, -1, -1):
        if response[start] == ']':
            depth += 1
        elif response[start] == '[':
            depth -= 1
            if depth == 0:
                return _decode_grid(response[start : end + 1]) or []
    return []
