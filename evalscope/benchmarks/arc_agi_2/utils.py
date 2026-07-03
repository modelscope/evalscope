# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import re
from typing import Any, Dict


def format_grid(grid: list) -> str:
    """Format a 2D grid as a JSON array string."""
    return json.dumps(grid)


def build_task_prompt(record: Dict[str, Any]) -> str:
    """Build the full prompt for an ARC-AGI-2 task from a dataset record."""
    fewshots = record['fewshots']
    question = record['question']

    parts = []
    parts.append(
        'You are given a series of input-output grid pairs as examples. '
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

    test_input = question[0]['input']
    parts.append(f'Test Input: {format_grid(test_input)}')
    parts.append('')
    parts.append('Provide the output grid as a JSON 2D array. Only output the JSON array, nothing else.')

    return '\n'.join(parts)


def parse_grid_from_response(response: str) -> list:
    """Extract a 2D grid (list of lists) from model response."""
    response = response.strip()

    # First try: direct JSON parse
    try:
        result = json.loads(response)
        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Second try: find content between ```json and ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    code_matches = re.findall(code_block_pattern, response, re.DOTALL)
    for match in code_matches:
        try:
            result = json.loads(match.strip())
            if isinstance(result, list) and all(isinstance(row, list) for row in result):
                return result
        except (json.JSONDecodeError, TypeError):
            continue

    # Third try: find JSON array pattern in text
    pattern = r'\[\s*\[.*?\]\s*\]'
    matches = re.findall(pattern, response, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, list) and all(isinstance(row, list) for row in result):
                return result
        except (json.JSONDecodeError, TypeError):
            continue

    return []
