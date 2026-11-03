import re
from typing import Dict, List, NamedTuple, Tuple

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
                text = text[: match.start()]
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


class _JavaToken(NamedTuple):
    text: str
    start: int
    end: int


_JAVA_LEXEME = re.compile(
    r'//[^\r\n]*|/\*.*?\*/|'  # Comments.
    r'"""(?:\\.|(?!""")[^\\])*"""|'  # Text blocks.
    r'"(?:\\.|[^"\\\r\n])*"|\'(?:\\.|[^\'\\\r\n])*\'|'  # String and character literals.
    r'[\w$]+|[^\s]',  # Identifiers and punctuation.
    re.DOTALL,
)
_JAVA_UNICODE_ESCAPE = re.compile(r'\\u+([0-9a-fA-F]{4})')


def _translate_java_unicode(source: str) -> Tuple[str, List[Tuple[int, int]]]:
    # Java translates eligible Unicode escapes before recognizing comments or literals (JLS 3.3).
    # Keep raw spans so editing structural tokens never rewrites literal contents.
    characters: List[str] = []
    spans: List[Tuple[int, int]] = []
    index = backslashes = 0
    previous_was_escape = False
    while index < len(source):
        start = index
        character = source[index]
        eligible = previous_was_escape or backslashes % 2 == 0
        previous_was_escape = False
        if character == '\\' and eligible and source[index + 1 : index + 2] == 'u':
            match = _JAVA_UNICODE_ESCAPE.match(source, index)
            if match is None:
                raise ValueError('Invalid Java Unicode escape')
            character = chr(int(match[1], 16))
            index = match.end()
            previous_was_escape = True
        else:
            index += 1
        backslashes = backslashes + 1 if character == '\\' else 0
        characters.append(character)
        spans.append((start, index))
    return ''.join(characters), spans


def _tokenize_java(source: str) -> List[_JavaToken]:
    """Read structural tokens with offsets into the original, untranslated source."""
    translated, spans = _translate_java_unicode(source)
    tokens = []
    for match in _JAVA_LEXEME.finditer(translated):
        text = match[0]
        if text.startswith(('//', '/*', '"', "'")):
            if text in {'"', "'"} or (text == '""' and translated.startswith('"""', match.start())):
                raise ValueError('Unterminated Java literal')
            continue
        if translated.startswith('/*', match.start()):
            raise ValueError('Unterminated Java comment')
        tokens.append(_JavaToken(text, spans[match.start()][0], spans[match.end() - 1][1]))
    return tokens


def _match_java_braces(tokens: List[_JavaToken]) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Return brace pairs, unclosed openings, and the depth before each token."""
    pairs = {}
    stack = []
    depths = []
    for index, token in enumerate(tokens):
        depths.append(len(stack))
        if token.text == '{':
            stack.append(index)
        elif token.text == '}':
            if not stack:
                raise ValueError('Unmatched Java closing brace')
            pairs[stack.pop()] = index
    return pairs, stack, depths


def _extract_java_test_body(tests: str) -> str:
    tokens = _tokenize_java(tests)
    if not tokens or tokens[0].text != '}' or tokens[-1].text != '}':
        raise ValueError('Expected a MultiPL-E Java method/class test suffix')
    middle = tokens[1:-1]
    _, stack, depths = _match_java_braces(middle)
    names = [token.text for token in middle]
    has_main = any(
        depth == 0 and names[index : index + 5] == ['public', 'static', 'void', 'main', '(']
        for index, depth in enumerate(depths)
    )
    if stack or not has_main:
        raise ValueError('MultiPL-E Java tests must contain a complete test main')
    return tests[tokens[0].end : tokens[-1].start]


def _assemble_java(code: str, tests: str) -> str:
    """Insert official tests into the solution class without trimming complete methods.

    Complete ``Problem``/``Main`` classes retain their methods and helper types. A legacy
    completion may leave just its class or its final method and class open. Other malformed
    structures remain a compile failure, rather than being truncated into a passing program.
    A separate ``Main`` entry point keeps ``Problem`` identifiers and class identity intact.
    This only locates class boundaries; javac remains responsible for Java syntax validation.
    """
    body = _extract_java_test_body(tests)
    try:
        tokens = _tokenize_java(code)
        pairs, stack, depths = _match_java_braces(tokens)
        targets = []
        for index, token in enumerate(tokens[:-1]):
            if token.text != 'class' or depths[index] != 0 or tokens[index + 1].text not in {'Problem', 'Main'}:
                continue
            # Annotation arguments in a class header can themselves contain brace pairs.
            parentheses = brackets = 0
            for opening in range(index + 2, len(tokens)):
                text = tokens[opening].text
                if text == '{' and parentheses == brackets == 0:
                    targets.append((tokens[index + 1].text, index, opening))
                    break
                parentheses += (text == '(') - (text == ')')
                brackets += (text == '[') - (text == ']')
        if len(targets) != 1:
            raise ValueError('Expected one top-level Problem or Main class')
        name, declaration, opening = targets[0]
        if opening in pairs:
            if stack:
                raise ValueError('Unclosed Java structure outside the solution class')
            insertion = tokens[pairs[opening]].start
            addition = f'\n{body}\n'
        else:
            if not stack or stack[0] != opening or len(stack) > 2:
                raise ValueError('Incomplete Java solution has unclosed nested blocks')
            # The official suffix supplies one method brace. Omit it when the method is closed.
            ending = ('}\n' if len(stack) == 2 else '') + body + '\n}'
            insertion = len(code)
            addition = f'\n{ending}\n'
        result = code[:insertion] + addition + code[insertion:]
        if name == 'Main':
            return result

        # The executor compiles Main.java. Keep Problem's name, including constructors and
        # references, but make its top-level declaration package-private if needed.
        header_start = parentheses = 0
        for index, token in enumerate(tokens[:declaration]):
            parentheses += (token.text == '(') - (token.text == ')')
            if parentheses == 0 and (
                (token.text == ';' and depths[index] == 0) or (token.text == '}' and depths[index] == 1)
            ):
                header_start = index + 1
        public = [token for token in tokens[header_start:declaration] if token.text == 'public']
        if len(public) > 1:
            raise ValueError('Repeated public modifier on the solution class')
        if public:
            modifier = public[0]
            # This modifier precedes the insertion, so its original offsets still apply.
            result = result[: modifier.start] + result[modifier.end :]
        return (
            result + '\nclass Main {\n'
            '    public static void main(String[] args) throws Exception { Problem.main(args); }\n'
            '}\n'
        )
    except ValueError:
        # Keep malformed model output in the normal compilation-failure scoring path.
        return f'#error Invalid MultiPL-E Java solution structure\n{code}\n{tests}'


def build_full_code(prediction: str, language: str, stop_tokens: List[str], tests: str) -> str:
    """
    Build executable full code from a model prediction:
    Java solutions retain complete methods and receive tests inside their solution class.
    Other languages use the existing extraction, stop-token, and test-suffix rules.

    Args:
        prediction: Raw model output.
        language: Dataset language identifier (e.g., 'humaneval-ts', 'mbpp-go', 'java').
        stop_tokens: Stop tokens to trim the solution code.
        tests: Dataset test code to append.

    Returns:
        Full code string ready for sandbox execution.
    """
    extract_lang, _ = normalize_languages(language)

    if extract_lang == 'java':
        code = default_extract_helper(prediction, extract_lang)
        return '\n'.join(IMPORT_HELPER.get(extract_lang, [])) + '\n' + _assemble_java(code, tests or '')

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

    return full_code
