import re
from typing import List, Tuple

from evalscope.utils.code_utils import (
    END_TOKENS,
    IMPORT_HELPER,
    default_extract_helper,
    extract_code_from_freeform_completion,
)


def normalize_languages(base: str) -> Tuple[str, str]:
    """
    Normalize dataset language identifiers into:
    - extract_lang: used for fenced-code blocks and extraction.
    - run_lang: runtime key used by sandbox executors.

    Args:
        base: Raw language identifier from dataset (e.g., 'humaneval-ts', 'mbpp-go', 'js', 'go_test.go').

    Returns:
        A tuple (extract_lang, run_lang).
    """
    # Normalize for extraction (code fences and end tokens)
    extract_map = {
        'ts': 'typescript',
        'js': 'js',
        'rkt': 'racket',
        'rb': 'ruby',
        'rs': 'rust',
        'jl': 'julia',
        'sh': 'bash',
        'd': 'd',
        'go_test.go': 'go',
        'pl': 'perl',
    }
    extract_lang = extract_map.get(base, base)

    # Map to sandbox runtime keys
    run_map = {
        'go': 'go_test',  # prefer test runner
        'go_test.go': 'go_test',
        'cs': 'csharp',
        'js': 'nodejs',
        'd': 'd_ut',
        'kotlin': 'kotlin_script',
        'verilog': 'verilog',
        'sh': 'bash',
        'rs': 'rust',
        'rb': 'ruby',
        'jl': 'julia',
        'rkt': 'racket',
        'jsnode': 'nodejs',
        'typescript': 'ts',
        'pl': 'perl',
    }
    run_lang = run_map.get(base, base)

    return extract_lang, run_lang


def trim_by_stop_tokens(text: str, stop_tokens: List[str]) -> str:
    """
    Trim text at the first occurrence of any stop token. Supports regex tokens with 're:' prefix.

    Args:
        text: Input text to trim.
        stop_tokens: List of stop tokens; regex tokens must start with 're:'.

    Returns:
        Trimmed text.
    """
    if not stop_tokens:
        return text
    for st in stop_tokens:
        if not st:
            continue
        if st.startswith('re:'):
            pattern = re.compile(st[3:].strip())
            match = pattern.search(text)
            if match:
                text = text[:match.start()]
        else:
            index = text.find(st)
            if index != -1:
                text = text[:index]
    return text


def stop_after_end_token(text: str, language: str) -> str:
    """
    Stop text after encountering an END_TOKEN defined for a language.

    Args:
        text: Input code text.
        language: Normalized extract language.

    Returns:
        Text truncated to include the first matched END_TOKEN (inclusive).
    """
    tokens = END_TOKENS.get(language, [])
    for et in tokens:
        index = text.find(et)
        if index != -1:
            return text[:index] + et
    return text


def remove_main(code: str, language: str) -> str:
    """
    Remove language-specific main entrypoint from code when needed.

    Args:
        code: Code snippet.
        language: Normalized extract language.

    Returns:
        Code without main entrypoint if detected.
    """
    main_tokens: List[str] = []
    if language == 'd':
        main_tokens = ['void main']
    elif language == 'csharp':
        main_tokens = ['public static void Main']
    for token in main_tokens:
        index = code.find(token)
        if index != -1:
            return code[:index]
    return code


def postprocess_full_code(code: str, language: str) -> str:
    """
    Post-process constructed full code for language-specific adjustments.

    - go: consolidate packages and imports into a single header, then place code body.

    Args:
        code: Full code including imports/helpers and solution.
        language: Normalized extract language.

    Returns:
        Post-processed code.
    """
    if language == 'go':
        # Collect all package declarations
        packages = set(re.findall(r'package\s+(\w+)', code))
        # Strip package lines from body
        code_body = re.sub(r'package\s+\w+\s*', '', code)

        # Collect imports: single and multi-form
        single_imports = re.findall(r'import\s+(".*?")', code_body)
        multi_block = re.findall(r'import\s*\((.*?)\)', code_body, flags=re.DOTALL)
        multi_imports: List[str] = []
        for blk in multi_block:
            multi_imports.extend([ln.strip() for ln in blk.split('\n') if ln.strip()])

        imports = {imp for imp in (single_imports + multi_imports) if imp}
        # Strip import lines from body
        code_body = re.sub(r'import\s+(".*?")', '', code_body)
        code_body = re.sub(r'import\s*\((.*?)\)', '', code_body, flags=re.DOTALL)

        # Rebuild header: packages + imports
        pkg_hdr = '\n'.join(f'package {p}' for p in packages) if packages else ''
        imp_hdr = '\n'.join(f'import {imp}' for imp in sorted(imports))
        hdr = '\n\n'.join([s for s in [pkg_hdr, imp_hdr] if s])

        # Final code layout
        code = f'{hdr}\n\n{code_body}'.strip()

    return code


def build_full_code(prediction: str, language: str, stop_tokens: List[str], tests: str) -> str:
    """
    Build executable full code from a model prediction:
    - Extract fenced code by language.
    - Trim with stop tokens.
    - Remove language-specific main entrypoints.
    - Prepend import helpers.
    - Language-specific post-processing.
    - Append dataset tests.
    - Java: rename Problem to Main.

    Args:
        prediction: Raw model output.
        language: Dataset language identifier (e.g., 'humaneval-ts', 'mbpp-go', 'java').
        stop_tokens: Stop tokens to trim the solution code.
        tests: Dataset test code to append.

    Returns:
        Full code string ready for sandbox execution.
    """
    extract_lang, _ = normalize_languages(language)

    use_func_extractor_langs = {'typescript', 'go', 'perl', 'racket', 'lua', 'julia', 'd', 'js', 'php', 'r', 'ruby'}
    if extract_lang in use_func_extractor_langs:
        code = default_extract_helper(prediction, extract_lang)
        code = remove_main(code, extract_lang)
    else:
        code, _ = extract_code_from_freeform_completion(prediction, extract_lang, first_block_only=True)
        code = trim_by_stop_tokens(code, stop_tokens or [])
        code = remove_main(code, extract_lang)

    import_helper = IMPORT_HELPER.get(extract_lang, [])
    full_code = '\n'.join(import_helper) + '\n' + code

    full_code = postprocess_full_code(full_code, extract_lang)
    full_code = f'{full_code}\n{tests or ""}'

    if extract_lang == 'java':
        full_code = full_code.replace('class Problem', 'class Main')

    return full_code
