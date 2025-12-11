"""Utilities for extracting and post-processing code blocks from free-form LLM completions.

This module provides:
- Fenced and heuristic code block extraction with language detection.
- Normalization helpers to remove entry points or wrap/trim code for specific languages.
- Support for custom extraction logic via exec-safe context.
"""

import re
from enum import Enum
from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple

from evalscope.utils.logger import get_logger

Language = Literal['python', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'php', 'csharp', 'bash', 'typescript', 'sql',
                   'rust', 'cuda', 'lua', 'r', 'perl', 'd_ut', 'ruby', 'scala', 'julia', 'pytest', 'junit',
                   'kotlin_script', 'jest', 'verilog', 'python_gpu', 'lean', 'swift', 'racket']
# Language identifiers supported by extraction/postprocessing utilities.

NullableLang = Language | Literal['']

logger = get_logger()

IMPORT_HELPER = {
    # Common imports per language to aid snippet compilation in sandbox contexts.
    # Note: Not automatically injected; used by callers where needed.
    'python': [
        'import math',
        'import re',
        'import sys',
        'import copy',
        'import datetime',
        'import itertools',
        'import collections',
        'import heapq',
        'import statistics',
        'import functools',
        'import hashlib',
        'import numpy',
        'import numpy as np',
        'import string',
        'from typing import *',
        'from collections import *',
    ],
    'cpp': [
        'using namespace std;',
        '#include<optional>',
        '#include<cassert>',
        '#include<stdlib.h>',
        '#include<algorithm>',
        '#include<cmath>',
        '#include<math.h>',
        '#include<numeric>',
        '#include<stdio.h>',
        '#include<vector>',
        '#include<set>',
        '#include<map>',
        '#include<queue>',
        '#include<stack>',
        '#include<list>',
        '#include<deque>',
        '#include<boost/any.hpp>',
        '#include<string>',
        '#include<climits>',
        '#include<cstring>',
        '#include<iostream>',
        '#include<sstream>',
        '#include<fstream>',
    ],
    'java': [
        'import java.util.*;',
        'import java.lang.reflect.*;',
        'import org.javatuples.*;',
        'import java.security.*;',
        'import java.math.*;',
        'import java.io.*;',
        'import java.util.stream.*;',
    ],
    'csharp': [
        'using System;',
        'using System.Numerics;',
        'using System.Diagnostics;',
        'using System.Collections.Generic;',
        'using System.Linq;',
        'using System.Text;',
        'using System.Security.Cryptography;',
        'using System.Collections.Generic;',
    ],
    'go': ["import (\"fmt\")"],
    'd': ['import std.array;', 'import std.algorithm;']
}

END_TOKENS = {
    # End tokens used by some languages to heuristically detect block boundaries.
    'julia': ['\nend'],
    'lua': ['\nend'],
    'ruby': ['\nend'],
    'cpp': ['\n}'],
    'csharp': ['\n}'],
    'typescript': ['\n}'],
    'perl': ['\n}'],
    'r': ['\n}'],
    'd': ['\n}'],
    'go': ['\n}'],
    'js': ['\n}'],
    'php': ['\n}'],
}


class ExtractedType(Enum):
    Fenced = 'fenced'
    IncompleteFenced = 'incomplete_fenced'
    Heuristic = 'heuristic'
    Empty = 'empty'


class CodeBlock(BaseModel):
    priority: int
    language: str
    code: str


fenced_code_block_pattern = re.compile(
    # Starting with three backticks and optional language identifier
    r'```([^\n]*)\n'
    r'(.*?)'  # Non-greedy capture of the content
    r'\n\s*```',  # Ending with three backticks
    re.DOTALL | re.MULTILINE
)

incomplete_fenced_code_block_pattern = re.compile(
    # Starting with three backticks and optional language identifier
    r'```([^\n]*)\n'
    r'(.*)',  # Greedy capture of the content
    re.DOTALL | re.MULTILINE
)

language_to_aliases = {
    'python': ['python', 'Python', 'py', 'Python3', 'python3', 'PY'],
    'cpp': ['cpp', 'c++', 'C++', 'Cpp', 'CPP'],
    'nodejs': ['javascript', 'Javascript', 'JavaScript', 'JS', 'js'],
    'go': ['go', 'Go'],
    'java': ['java', 'Java'],
    'csharp': ['csharp', 'c#', 'C#'],
    'bash': ['bash', 'Bash', 'BASH', 'sh', 'shell'],
    'typescript': ['typescript'],
    'rust': ['rust', 'Rust', 'rs'],
    'sql': ['sql', 'SQL', 'Sql'],
    'd': ['D', 'd'],
    'julia': ['julia', 'Julia', 'jl'],
    'lua': ['lua', 'Lua'],
    'php': ['php', 'PHP'],
    'perl': ['perl', 'Perl', 'PERL'],
    'r': ['R', 'r'],
    'ruby': ['ruby', 'Ruby'],
    'scala': ['scala', 'Scala'],
    'kotlin': ['kotlin', 'Kotlin'],
    'c': ['c', 'C'],
    'html': ['html', 'Html', 'HTML'],
    'javascript': ['javascript', 'Javascript', 'JavaScript'],
    'verilog': ['verilog', 'Verilog', 'VERILOG'],
    'racket': ['racket'],
    'swift': ['swift'],
}

aliases_to_language_tiled = {v: k for k, vs in language_to_aliases.items() for v in vs}


# code extraction
def extract_fenced_code(completion: str) -> List[CodeBlock]:
    """Extract complete fenced code blocks from a completion.

    A fenced block is delimited by triple backticks with an optional language tag:
    ```lang
    code
    ```

    Returns a list of CodeBlock ordered by appearance with priority=30.
    """
    code_matches = re.findall(fenced_code_block_pattern, completion)
    results = []
    for m in code_matches:
        lang = aliases_to_language_tiled.get(m[0].strip(), '')
        results.append(CodeBlock(priority=30, language=lang, code=m[1]))
    return results


def adjust_code_block(code_blocks: List[CodeBlock], language: str) -> List[CodeBlock]:
    """Fix language tag placement issues and adjust blocks to the target language.

    Some models place the language alias as the first line inside the fence rather than in the fence tag.
    If a block has empty language and its first line matches an alias for `language`, the alias line is removed
    and the block's language is set to `language`.
    """
    if language == '' or language not in language_to_aliases:
        return code_blocks
    ret = []
    for block in code_blocks:
        lines = block.code.splitlines()
        if block.language == '' and lines and lines[0].strip() in language_to_aliases[language]:
            block.language = language
            block.code = '\n'.join(lines[1:])
        ret.append(block)
    return ret


# code extraction


def extract_incomplete_fenced_code(completion: str) -> List[CodeBlock]:
    """Extract incomplete fenced code blocks that start with ``` but do not close.

    Useful for handling truncated outputs. Returns CodeBlock with priority=20.
    """
    code_matches = re.findall(incomplete_fenced_code_block_pattern, completion)
    results = []
    for m in code_matches:
        lang = aliases_to_language_tiled.get(m[0].strip(), '')
        results.append(CodeBlock(priority=20, language=lang, code=m[1]))
    return results


def extract_heuristic_code(completion: str, language: NullableLang = '') -> List[CodeBlock]:
    """Extract code via simple heuristics when fenced blocks are missing.

    Currently supports:
    - python: detect function/class definitions and bodies.
    - sql: detect SELECT/CTE queries.
    - bash: return non-empty lines as a single block.

    Returns language-specific CodeBlock with priority=10 or an empty list.
    """

    def extract_py(text):
        code = '\n'.join([line for line in text.split('\n') if line.strip() != '']) + '\n'

        pattern_py = '(?:^(?:import|from|#)[^\n]+\n)*' \
            '^(?:def|class) [^\n]+\n' \
            r'(?:\s+[^\n]+\n)+'  # class or function body
        matches = re.findall(pattern_py, code, re.M)
        return matches

    def extract_sql(text):
        code = '\n'.join([line for line in text.split('\n') if line.strip() != '']) + '\n'

        pattern_sql = r'^\s*(?:select|with\s[^\n]+as)[^;]*'
        matches = re.findall(pattern_sql, code, re.M | re.IGNORECASE)
        return matches

    def extract_bash(text):
        code = '\n'.join([line for line in text.split('\n') if line.strip() != '']) + '\n'
        return code

    if language == 'python':
        return [CodeBlock(priority=10, language='python', code=m) for m in extract_py(completion)]
    elif language == 'sql':
        return [CodeBlock(priority=10, language='sql', code=m) for m in extract_sql(completion)]
    elif language == 'bash':
        return [CodeBlock(priority=10, language='bash', code=extract_bash(completion))]
    else:
        return []


def extract_custom_code(completion: str, custom_logic: str) -> List[CodeBlock]:
    """Run custom extraction logic provided as a Python string.

    The execution context includes:
    - CodeBlock class
    - completion (str)
    - submit_code_blocks(cb_list) to append CodeBlock instances
    - extract_fenced_code, extract_heuristic_code helpers

    Note: The caller is responsible for the safety of `custom_logic`.
    """
    blocks = []

    def submit(cbs):
        for cb in cbs:
            assert isinstance(cb, CodeBlock), 'extracted code type must be class CodeBlock'
            blocks.append(cb)

    context = {
        'CodeBlock': CodeBlock,
        'completion': completion,
        'submit_code_blocks': submit,
        'extract_fenced_code': extract_fenced_code,
        'extract_heuristic_code': extract_heuristic_code,
    }
    exec(custom_logic, context)
    logger.info(f'got {len(blocks)} custom code blocks')
    return blocks


def filter_language(blocks: List[CodeBlock], language: NullableLang) -> List[CodeBlock]:
    """Filter CodeBlock list by exact language match."""
    return [b for b in blocks if b.language == language]


def trim_code_entrypoint(completion: str, language: NullableLang = ''):
    """Trim common entry points like main functions for some languages.

    Implement or remove if redundant with postprocess_completion/remove_entripoints.
    """
    ...


def default_extract_helper(completion: str, language: NullableLang = '', custom_extract_logic: Optional[str] = None):
    """Default strategy to obtain one code string from a completion.

    Order of extraction:
    1) Fenced code blocks
    2) Language-specific heuristics
    3) Incomplete fenced blocks
    4) Optional custom logic

    Selection rule:
    - Choose blocks with max priority; prefer target `language` if available; otherwise take the first.
    """
    code_blocks = extract_fenced_code(completion)
    code_blocks += extract_heuristic_code(completion, language)
    code_blocks += extract_incomplete_fenced_code(completion)
    if custom_extract_logic is not None:
        code_blocks += extract_custom_code(completion, custom_extract_logic)
    if len(code_blocks) == 0:
        return ''

    max_priority = max([cb.priority for cb in code_blocks])
    code_blocks = [cb for cb in code_blocks if cb.priority == max_priority]

    target_blocks = filter_language(code_blocks, language)
    if len(target_blocks) > 0:
        return target_blocks[0].code
    return code_blocks[0].code


def remove_entripoints(code, language: NullableLang = ''):
    """Remove typical entry points to keep only reusable logic.

    Examples:
    - python: strip `if __name__ == "__main__": ...`
    - cpp: strip `int main()`
    - go: remove `package main`
    - strip sections starting at `# Example usage`
    """
    if language == 'python':
        if 'if __name__ == \"__main__\":' in code:
            next_line = code.index('if __name__ == \"__main__\":')
            code = code[:next_line].strip()
    elif language == 'cpp':
        if 'int main()' in code:
            next_line = code.index('int main()')
            code = code[:next_line].strip()
    elif language == 'go':
        # Remove package main
        code = code.replace('package main', '')
    if '# Example usage' in code:
        next_line = code.index('# Example usage')
        code = code[:next_line].strip()
    return code


# compatible function for evals/evals/elsuite/utils/coding_evaluation/utils_coding/extract_code_from_freeform_completion
def extract_code_from_freeform_completion(
    completion: str, language: NullableLang = '', first_block_only=False, **kwargs
) -> Tuple[str, str]:
    """ Returns: (code, extracted_type)
    """
    extracted_type = ExtractedType.Empty  # initialize to empty case

    # step1. match the complete fenced block
    code_blocks = extract_fenced_code(completion)

    if kwargs.get('is_fewshot_task') is True:
        first_sp_block_idx = next((i for i, block in enumerate(code_blocks) if block.language == language), -1)
        first_un_block_idx = next((i for i, block in enumerate(code_blocks) if block.language == ''), -1)
        first_block_idx = first_un_block_idx if first_sp_block_idx == -1 else first_sp_block_idx
        if first_block_idx != -1:
            code_blocks = code_blocks[:first_block_idx + 1]

        logger.debug(f'select first code block for fewshot task: {code_blocks}')

    # drop the blocks which the language tag different with target programming language
    if kwargs.get('exactly_match') is True and language:
        other_tag = set(sum([v for k, v in language_to_aliases.items() if k != language], []))
        code_blocks = [b for b in code_blocks if b.language not in other_tag]

    if code_blocks:
        extracted_type = ExtractedType.Fenced

    # step2. if no complete fenced block found, then match the incomplete fenced block
    if len(code_blocks) == 0:
        code_blocks = extract_incomplete_fenced_code(completion)
        if code_blocks:
            extracted_type = ExtractedType.IncompleteFenced

    # step3. if no incomplete fenced block found, try heuristic method to extract code
    if len(code_blocks) == 0:
        code_blocks = extract_heuristic_code(completion, language)
        if code_blocks:
            extracted_type = ExtractedType.Heuristic

    if kwargs.get('code_block_idx') is not None:
        try:
            completion = code_blocks[kwargs['code_block_idx']].code.replace('\r', '')
        except Exception:
            completion = ''
    elif first_block_only:
        if code_blocks:
            completion = code_blocks[0].code.replace('\r', '')
        else:
            completion = ''
    else:
        completion = '\n\n'.join([b.code for b in code_blocks]).replace('\r', '')

    if language == 'python':

        if kwargs.get('remove_asserts') is True:
            # remove assert statements
            lines = []
            for line in completion.split('\n'):
                if line.startswith('assert '):
                    continue
                else:
                    lines.append(line)
            completion = '\n'.join(lines)

        if 'if __name__ == \"__main__\":' in completion:
            next_line = completion.index('if __name__ == \"__main__\":')
            completion = completion[:next_line].strip()
    elif language == 'cpp':
        if 'int main()' in completion:
            next_line = completion.index('int main()')
            completion = completion[:next_line].strip()
    elif language == 'java':
        # Add class Solution before signature
        if 'public class Main {\n' in completion:
            completion = completion.replace('public class Main {\n', 'class Solution {\n')
            completion = completion.replace('public static void main(String[] args)', '')
        if 'class Solution' not in completion:
            for line in completion.split('\n'):
                if kwargs.get('entry_point') and kwargs.get('entry_point') in line:
                    completion = completion.replace(line, 'class Solution {\n' + line)
                    completion += '\n}'
                    break
        # Add import statements
        for line in kwargs.get('declaration', '').split('\n'):
            if 'import' in line:
                completion = line + '\n' + completion
    elif language == 'go':
        # Remove package main
        completion = completion.replace('package main', '')
    if '# Example usage' in completion:
        next_line = completion.index('# Example usage')
        completion = completion[:next_line].strip()

    return (completion, extracted_type.value)


def extract_code_from_freeform_completion_v2(
    completion: str,
    language: NullableLang = '',
    first_block_only=False,
    no_removal=False,
    **kwargs
) -> Tuple[str, str]:
    """
    Arguments:
    - kwargs:
        - inner_function_only(bool): used for language like c#, java, etc.

    Returns: (code, extracted_type)

    Since == autoeval-v5

    - Modified the logic for removing python main execution part

    - Adapted to llama3's abnormal Code block format
    """
    completion_bk = completion  # backup the input
    extracted_type = ExtractedType.Empty  # initialize to empty case

    # step0. preprocess
    completion = completion.replace('```\n```', '```')  # solve llama3 error format

    # step1. match the complete fenced block
    code_blocks = extract_fenced_code(completion)
    code_blocks = adjust_code_block(code_blocks, language)  # solve llama3 error format

    if kwargs.get('is_fewshot_task') is True:
        first_sp_block_idx = next((i for i, block in enumerate(code_blocks) if block.language == language), -1)
        first_un_block_idx = next((i for i, block in enumerate(code_blocks) if block.language == ''), -1)
        first_block_idx = first_un_block_idx if first_sp_block_idx == -1 else first_sp_block_idx
        if first_block_idx != -1:
            code_blocks = code_blocks[:first_block_idx + 1]

        logger.debug(f'select first code block for fewshot task: {code_blocks}')

    # drop the blocks which the language tag different with target programming language
    if kwargs.get('exactly_match') is True and language:
        target_tag = language_to_aliases.get(language, [])
        code_blocks = [b for b in code_blocks if b.language in target_tag]

    if code_blocks:
        extracted_type = ExtractedType.Fenced

    # step2. if no complete fenced block found, then match the incomplete fenced block
    if len(code_blocks) == 0:
        code_blocks = extract_incomplete_fenced_code(completion)
        if code_blocks:
            extracted_type = ExtractedType.IncompleteFenced

    # step3. if no incomplete fenced block found, try heuristic method to extract code
    if len(code_blocks) == 0:
        code_blocks = extract_heuristic_code(completion, language)
        if code_blocks:
            extracted_type = ExtractedType.Heuristic

    if kwargs.get('code_block_idx') is not None:
        try:
            completion = code_blocks[kwargs['code_block_idx']].code.replace('\r', '')
        except Exception:
            completion = ''
    elif first_block_only:
        if code_blocks:
            completion = code_blocks[0].code.replace('\r', '')
        else:
            completion = ''
    else:
        completion = '\n\n'.join([b.code for b in code_blocks]).replace('\r', '')

    is_ut = kwargs.get('is_ut') is True
    if not is_ut:
        completion = postprocess_completion_v2(completion, language, no_removal, completion_bk, **kwargs)

    if '# Example usage' in completion:
        next_line = completion.index('# Example usage')
        completion = completion[:next_line].strip()

    return (completion, extracted_type.value)


def postprocess_completion_v2(completion: str, language: str, no_removal: bool, completion_bk: str, **kwargs) -> str:
    inner_function_only = kwargs.get('inner_function_only') is True

    if language == 'python':
        lines = completion.splitlines()
        idx = None
        for i, line in enumerate(lines):
            if '__name__' in line and '__main__' in line:
                idx = i
                break
        if idx is not None:
            lines = lines[:idx]
        completion = '\n'.join(lines)

        if kwargs.get('remove_asserts') is True:
            lines = []
            for line in completion.splitlines():
                if not line.startswith('assert '):
                    lines.append(line)
            completion = '\n'.join(lines)

    elif language in ['cpp', 'c']:
        if 'int main()' in completion:
            next_line = completion.index('int main()')
            completion = completion[:next_line].strip()
    elif language == 'java':
        if inner_function_only:
            pattern = r'(public|private|protected)\s+(static\s+)(.*?)\((.*?)\)\s*{'
            body = find_inner_function_body(pattern, completion)
            if body is not None:
                completion = completion[body[0]:body[1]]
        else:
            # Add class Solution before signature
            if 'public class Main {\n' in completion:
                completion = completion.replace('public class Main {\n', 'class Solution {\n')
                completion = completion.replace('public static void main(String[] args)', '')
            # remove `public` of class `Solution`
            if 'public class Solution {' in completion:
                completion = completion.replace('public class Solution {', 'class Solution {')
            if 'class Solution' not in completion:
                for line in completion.split('\n'):
                    if kwargs.get('entry_point') and kwargs.get('entry_point') in line:
                        completion = completion.replace(line, 'class Solution {\n' + line)
                        completion += '\n}'
                        break
            # Add import statements
            for line in kwargs.get('declaration', '').split('\n'):
                if 'import' in line:
                    completion = line + '\n' + completion
    elif language == 'go':
        # Generally remove the `package main` statement, but some datasets should not remove it, such as mbxp
        if not no_removal:
            completion = completion.replace('package main', '')

        # Delete the main function from completion, if exists.
        pattern = r'func\s+main\(.*?\)\s*{'
        body = find_inner_function_body(pattern, completion)
        if body is not None:
            completion = completion[:body[0]] + completion[body[1]:]
    elif language == 'scala':
        # Extract the part wrapped in object X { ... }, generally it's a function
        pat = r'object\s+\w+(\s+extends\s+\w+)?\s*\n*\{(.*)\}'
        r = re.findall(pat, completion, re.DOTALL | re.MULTILINE)
        if r:
            completion = r[0][1]
    elif language == 'verilog':
        # Extract the content between the semicolon and endmodule in module X (X, X); ... endmodule, including endmodule
        pat = r'module\s+\w+\s+\((.*?)\);(.*?)endmodule'
        r = re.findall(pat, completion, re.DOTALL | re.MULTILINE)
        if r:
            completion = r[0][1] + '\nendmodule'
        if completion == '':
            # if we cannot extract any code block, return the unacted input
            completion = completion_bk
    elif language == 'csharp':
        # Extract function body part inside class
        if inner_function_only:
            pattern = r'(public|private|protected|internal)\s+(static\s+)(.*?)\((.*?)\)\s*{'
            body = find_inner_function_body(pattern, completion)
            if body is not None:
                completion = completion[body[0]:body[1]]
    elif language == 'kotlin':
        # Delete the main function from completion, if exists.
        pattern = r'fun\s+main\(.*?\)\s*{'
        body = find_inner_function_body(pattern, completion)
        if body is not None:
            completion = completion[:body[0]] + completion[body[1]:]
    return completion


def trim_till_first_function(code, language):
    # Regex patterns to find the start of a function
    if language == 'python':
        pattern = r'\bdef\s+\w+\s*\((?:[^()]|\n)*\)\s*->?\s*[\w\[\],\s]*:'
        # Python uses indentation, not brackets
        open_bracket, close_bracket = ':', None
    elif language in ['golang', 'go']:
        pattern = r'\bfunc\s+\w+\s*\([^)]*\)\s*(\[\w+\]|\*?\w*)?\s*{'
        open_bracket, close_bracket = '{', '}'
    elif language == 'typescript':
        pattern = r'\bfunction\s+\w+\s*\([^)]*\)\s*[:\w\s]*{'
        open_bracket, close_bracket = '{', '}'
    else:
        raise ValueError('Unsupported language')

    # Find the start of the first function
    match = re.search(pattern, code)
    if not match:
        return ''  # No function found

    if close_bracket:
        # Count brackets to find the end
        start_index = match.start()
        end_index = start_index
        bracket_count = 0
        in_string = False
        escape = False
        while end_index < len(code):
            char = code[end_index]
            if char in ('"', "'"):
                # Handle strings
                if not in_string:
                    in_string = True
                    string_delimiter = char
                elif char == string_delimiter and not escape:
                    in_string = False
            elif not in_string:
                if char == open_bracket:
                    bracket_count += 1
                elif char == close_bracket:
                    bracket_count -= 1
                    if bracket_count == 0:
                        break
            # Handle escape characters in strings
            escape = char == '\\' and not escape
            end_index += 1

        return code[:end_index + 1]
    else:
        # For Python, use indentation levels
        lines = code[match.end():].splitlines()
        first_line_indent = len(lines[0]) - len(lines[0].lstrip())
        function_code = code[:match.end()]
        for line in lines[1:]:
            indent = len(line) - len(line.lstrip())
            if line.strip() and indent <= first_line_indent:
                break
            function_code += '\n' + line

        return function_code


def find_java_public_class_name(java_code: str) -> str:
    """
    Finds and returns the name of the public class in a given Java source code string.
    If no public class is found, returns None.

    Args:
    java_code (str): A string containing Java source code.

    Returns:
    str or None: The name of the public class if found, otherwise None.
    """
    pattern = r'\bpublic\s+(abstract\s+|final\s+)?class\s+(\w+)'
    match = re.search(pattern, java_code)
    if match:
        return match.group(2)
    else:
        return None


def find_inner_function_body(signature_pattern: str, completion: str) -> Optional[Tuple[int, int]]:
    """
    Finds inner function body inside some class/namespace block.
    Used for language like: c#, java, etc.

    Args:
    signature_pattern (str): Function signature pattern, includes the left curly brackets,
        used to find the starting position of function.

    Returns:
    Tuple[int, int] or None: The function body indices if exists, otherwise None.
    """
    matches = re.search(signature_pattern, completion, re.DOTALL | re.MULTILINE)
    if matches is None:
        return None
    brackets_count = 1
    idx = None
    for idx in range(matches.end(), len(completion)):
        if completion[idx] == '{':
            brackets_count += 1
        elif completion[idx] == '}':
            brackets_count -= 1

        if brackets_count == 0:
            break
    if idx is None or brackets_count != 0:
        return None
    return (matches.start(), idx + 1)
