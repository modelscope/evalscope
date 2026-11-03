"""Assemble complete Java solutions with MultiPL-E's completion-oriented tests."""

import re
from typing import Dict, List, NamedTuple, Tuple


class _Token(NamedTuple):
    text: str
    start: int
    end: int


_LEXEME = re.compile(
    r'//[^\r\n]*|/\*.*?\*/|"""(?:\\.|(?!""")[^\\])*"""|'
    r'"(?:\\.|[^"\\\r\n])*"|\'(?:\\.|[^\'\\\r\n])*\'|[\w$]+|[^\s]',
    re.DOTALL,
)
_UNICODE_ESCAPE = re.compile(r'\\u+([0-9a-fA-F]{4})')


def _translate_unicode(source: str) -> Tuple[str, List[Tuple[int, int]]]:
    # Java translates eligible Unicode escapes before recognizing comments or literals (JLS 3.3).
    # Keep raw spans so editing structural tokens never rewrites literal contents.
    characters: List[str] = []
    spans: List[Tuple[int, int]] = []
    index = backslashes = 0
    escaped = False
    while index < len(source):
        start = index
        character = source[index]
        eligible = escaped or backslashes % 2 == 0
        escaped = False
        if character == '\\' and eligible and source[index + 1 : index + 2] == 'u':
            match = _UNICODE_ESCAPE.match(source, index)
            if match is None:
                raise ValueError('Invalid Java Unicode escape')
            character = chr(int(match[1], 16))
            index = match.end()
            escaped = True
        else:
            index += 1
        backslashes = backslashes + 1 if character == '\\' else 0
        characters.append(character)
        spans.append((start, index))
    return ''.join(characters), spans


def _tokens(source: str) -> List[_Token]:
    translated, spans = _translate_unicode(source)
    tokens = []
    for match in _LEXEME.finditer(translated):
        text = match[0]
        if text.startswith(('//', '/*', '"', "'")):
            if text in {'"', "'"} or (text == '""' and translated.startswith('"""', match.start())):
                raise ValueError('Unterminated Java literal')
            continue
        if translated.startswith('/*', match.start()):
            raise ValueError('Unterminated Java comment')
        tokens.append(_Token(text, spans[match.start()][0], spans[match.end() - 1][1]))
    return tokens


def _braces(tokens: List[_Token]) -> Tuple[Dict[int, int], List[int], List[int]]:
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


def _test_body(tests: str) -> str:
    tokens = _tokens(tests)
    if not tokens or tokens[0].text != '}' or tokens[-1].text != '}':
        raise ValueError('Expected a MultiPL-E Java method/class test suffix')
    middle = tokens[1:-1]
    _, stack, depths = _braces(middle)
    names = [token.text for token in middle]
    has_main = any(
        depth == 0 and names[index : index + 5] == ['public', 'static', 'void', 'main', '(']
        for index, depth in enumerate(depths)
    )
    if stack or not has_main:
        raise ValueError('MultiPL-E Java tests must contain a complete test main')
    return tests[tokens[0].end : tokens[-1].start]


def _apply_edits(source: str, edits: List[Tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def assemble_java(code: str, tests: str) -> str:
    """Insert official tests into the solution class without trimming complete methods.

    Complete ``Problem``/``Main`` classes retain their methods and helper types. A legacy
    completion may leave just its class or its final method and class open. Other malformed
    structures remain a compile failure, rather than being truncated into a passing program.
    A separate ``Main`` entry point keeps ``Problem`` identifiers and class identity intact.
    """
    body = _test_body(tests)
    try:
        tokens = _tokens(code)
        pairs, stack, depths = _braces(tokens)
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
            edits = [(insertion, insertion, f'\n{body}\n')]
        else:
            if not stack or stack[0] != opening or len(stack) > 2:
                raise ValueError('Incomplete Java solution has unclosed nested blocks')
            # The official suffix supplies one method brace. Omit it when the method is closed.
            ending = ('}\n' if len(stack) == 2 else '') + body + '\n}'
            edits = [(len(code), len(code), f'\n{ending}\n')]
        if name == 'Main':
            return _apply_edits(code, edits)

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
        edits.extend((token.start, token.end, '') for token in public)
        result = _apply_edits(code, edits)
        return (
            result + '\nclass Main {\n'
            '    public static void main(String[] args) throws Exception { Problem.main(args); }\n'
            '}\n'
        )
    except ValueError:
        # Keep malformed model output in the normal compilation-failure scoring path.
        return f'#error Invalid MultiPL-E Java solution structure\n{code}\n{tests}'
