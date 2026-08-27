# flake8: noqa: E501
import re
from typing import List, Optional, Union

from evalscope.api.evaluator import Choices, Target, TaskState

FEW_SHOT_TEMPLATE = r"""Here are some examples of how to answer similar questions:

{fewshot}

""".lstrip()

SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
""".strip()

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()

MULTIPLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question where multiple answers may be correct. The entire content of your response should be of the following format: 'ANSWER: [LETTERS]' (without quotes) where [LETTERS] is one or more of {letters}.

{question}

{choices}
""".strip()

MULTIPLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question where multiple answers may be correct. The last line of your response should be of the following format: 'ANSWER: [LETTERS]' (without quotes) where [LETTERS] is one or more of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()

CHINESE_FEW_SHOT_TEMPLATE = r"""以下是一些示例问题：

{fewshot}

""".lstrip()

CHINESE_SINGLE_ANSWER_TEMPLATE = r"""回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。

问题：{question}
选项：
{choices}
""".lstrip()

CHINESE_SINGLE_ANSWER_TEMPLATE_COT = r"""回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}
""".lstrip()

CHINESE_MULTIPLE_ANSWER_TEMPLATE = r"""回答下面的多项选择题，请选出其中的所有正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTERS]"（不带引号），其中 [LETTERS] 是 {letters} 中的一个或多个。
问题：{question}
选项：
{choices}
""".lstrip()

CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT = r"""回答下面的多项选择题，请选出其中的所有正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTERS]"（不带引号），其中 [LETTERS] 是 {letters} 中的一个或多个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}
""".lstrip()


def unshuffle_choices(choices: Choices) -> Choices:
    # `sorted` returns `list[Choice]`, but for consistency we wrap this back
    # into a `Choices` object
    return Choices(sorted(choices, key=lambda choice: choice.original_position))


def answer_options(choices: Union[Choices, List[str]]) -> str:
    r"""
    Returns the `choices` formatted as a multiple choice question, e.g.:

    ["choice 1", "choice 2", "choice 3"] ->
        "A) choice 1\nB) choice 2\nC) choice 3"
    """
    if isinstance(choices, list):
        choices = Choices(choices)

    indexes = list(range(len(choices)))

    return '\n'.join([f'{answer_character(i)}) {choices[j].value}' for i, j in enumerate(indexes)])


def format_letter_choices(choices: Union[Choices, List[str]]) -> str:
    """
    Returns the `choices` formatted as a letter list, e.g.:

    ["choice 1", "choice 2", "choice 3"] ->
        "A,B,C"
    """
    if isinstance(choices, list):
        choices = Choices(choices)

    indexes = list(range(len(choices)))

    return ','.join([f'{answer_character(i)}' for i in indexes])


def prompt(question: str, choices: Union[Choices, List[str]], template: str, fewshot: Optional[str] = None) -> str:
    if isinstance(choices, list):
        choices = Choices(choices)

    choices_text = answer_options(choices)
    letters = format_letter_choices(choices)
    if not fewshot:
        return template.format(
            choices=choices_text,
            letters=letters,
            question=question,
        )
    else:
        return template.format(
            choices=choices_text,
            letters=letters,
            question=question,
            fewshot=fewshot,
        )


def format_example(
    question: str,
    choices: Choices,
    answer: Target,
) -> str:
    """Format a single example for few-shot learning.

    Args:
        question (str): The question text.
        choices (list[str]): The list of choices.
        answer (list[str]): The correct answers.

    Returns:
        str: Formatted example string.
    """
    choices_text = answer_options(choices)
    return f'{question}\n{choices_text}\nANSWER: {answer.text}'


# The answer marker itself.  Locating markers separately from the label keeps a greedy label
# pattern from swallowing the next marker, which would hide it from the scan below.
_ANSWER_MARKER_RE = re.compile(r'(?i)ANSWER\s*:\s*\**\s*')
_ANSWER_MARKER_ZH_RE = re.compile(r'答案\s*[:：]\s*\**\s*')

# A label the model may wrap the way the options are printed, optionally listing several labels:
# '(A)', '[A]', '(A, C)', '(A/C)'.  Only label-shaped content is accepted, so bracketed prose
# such as '(see the diagram above)' and an echoed '[LETTER]' placeholder stay unparseable.
_BRACKETED_LABEL_RE = re.compile(r'[\(\[（【]\s*([A-Za-z\d](?:\s*[,，/、]\s*[A-Za-z\d])*)\s*[\)\]）】]')

_PLAIN_LABEL_RE = re.compile(r'([A-Za-z\d][A-Za-z\d ,/、]*)')
_PLAIN_LABEL_ZH_RE = re.compile(r'([A-Za-z0-9][A-Za-z0-9,，]*)')

_LABEL_TOKEN_RE = re.compile(r'[A-Za-z\d]+')

# Words a model may place between two labels when listing several of them ('A and B').
_LABEL_CONNECTORS = {'and', 'or'}
_LABEL_CONNECTOR_RE = re.compile(r'\s+(?:and|or)\s+', re.IGNORECASE)


def _fallback_parse_answer(completion: str, allowed_options: set[str]) -> Optional[set[str]]:
    # Fallback to find the last upper case letter
    for letter in reversed(completion):
        if letter.isupper():
            # A letter that is not one of the sample's labels cannot be the model's choice, and
            # returning it would record an answer the model never gave.  Reporting no answer
            # scores the same (an invalid label never equals the target) and stays diagnosable.
            return {letter} if letter in allowed_options else None
    return None


def _is_label_word(word: str, allowed_options: set[str]) -> bool:
    return word in allowed_options or set(word).issubset(allowed_options)


def _label_prefix(capture: str, allowed_options: set[str]) -> str:
    """Leading part of a capture that holds nothing but labels of the current sample.

    Models routinely justify the choice in the same breath ('ANSWER: B, not C'), and a
    capture rejected as a whole loses the label with the prose: the reply then reaches
    `_fallback_parse_answer`, which answers with the last capital - typically a distractor
    named in that justification.  A connector between two labels ('A and B', 'A or B')
    separates list items and is stepped over; a connector anywhere else ends the answer.
    """
    tokens = list(_LABEL_TOKEN_RE.finditer(capture))
    end = 0
    for i, token in enumerate(tokens):
        word = token.group(0)
        if _is_label_word(word, allowed_options):
            end = token.end()
            continue
        # Skip a connector only when another label follows it, so prose such as
        # 'B, not C' or 'B and that is why' cannot swallow later labels.
        follows_label = end > 0
        precedes_label = i + 1 < len(tokens) and _is_label_word(tokens[i + 1].group(0), allowed_options)
        if follows_label and precedes_label and word.lower() in _LABEL_CONNECTORS:
            continue
        break
    return capture[:end]


def _last_labelled_answer(
    text: str,
    marker_re: re.Pattern,
    plain_label_re: re.Pattern,
    allowed_options: set[str],
) -> Optional[str]:
    """Label of the last answer marker that is actually followed by one.

    Reasoning models restate the required format and revise themselves mid-thought, so an
    earlier marker may carry an echoed placeholder or a choice the model went on to reject.
    """
    for marker in reversed(list(marker_re.finditer(text))):
        tail = text[marker.end():]
        for label_re in (_BRACKETED_LABEL_RE, plain_label_re):
            label = label_re.match(tail)
            if label is None:
                continue
            prefix = _label_prefix(label.group(1), allowed_options)
            if prefix:
                return prefix
    return None


def parse_answers(state: TaskState, multiple_correct: bool = False, completion: Optional[str] = None) -> set[str]:
    """
    Convenience function for extracting answers from the state output.

    The generated response must be in the format 'ANSWER: <answers>',
    otherwise we can't extract what the model thinks is "true". We can be a
    bit flexible whether these are "AB" vs "A,B" vs "A B".

    However, if the answer isn't in the expected format the model has
    failed in the task so we'll ultimately just mark it as incorrect

    Args:
        state: The task state holding the model output and the available choices.
        multiple_correct: Whether more than one choice may be correct.
        completion: Text to parse answers from. Defaults to the raw model completion;
            callers inside `extract_answer` should pass the filtered prediction so that
            configured filters (e.g. `remove_until`) actually take effect.
    """
    text = state.output.completion if completion is None else completion

    allowed_options = set(answer_character(i) for i in range(len(state.choices)))

    matched = _last_labelled_answer(text, _ANSWER_MARKER_RE, _PLAIN_LABEL_RE, allowed_options)

    if matched is None:
        return _fallback_parse_answer(text, allowed_options) or set()

    # Strip trailing period / full stop
    matched = matched.strip()
    matched = matched.rstrip('.')

    if multiple_correct:
        # Match must contain only the allowed choices
        # (may be separated by commas, slashes, spaces, the words 'and'/'or', or nothing at all)

        matched = _LABEL_CONNECTOR_RE.sub(',', matched)

        matched = matched.replace(' ', '')

        # The label patterns also accept full-width separators, which the split below would
        # otherwise keep inside the label and turn a valid multi-select answer into no answer.
        matched = matched.replace('，', ',').replace('、', ',').replace('/', ',')

        split_comma = set(matched.split(','))
        if split_comma.issubset(allowed_options):
            answers = split_comma
            return answers

        # 'AB,CD' also lists the labels one by one; split it into single characters.
        split_nothing = set(matched.replace(',', ''))
        if split_nothing.issubset(allowed_options):
            answers = split_nothing
            return answers

    else:
        # Match must contain a single letter in the allowed choices
        if matched in allowed_options:
            answers = {matched}
            return answers

    return set()


def parse_answers_zh(state: TaskState, multiple_correct: bool = False, completion: Optional[str] = None) -> set[str]:
    """
    Convenience function for extracting answers from the state output in Chinese format.

    The generated response must be in the format '答案：选项',
    otherwise we can't extract what the model thinks is "true". We can be a
    bit flexible whether these are "AB" vs "A,B" vs "A B".

    Args:
        state: The task state holding the model output and the available choices.
        multiple_correct: Whether more than one choice may be correct.
        completion: Text to parse answers from. Defaults to the raw model completion;
            callers inside `extract_answer` should pass the filtered prediction so that
            configured filters (e.g. `remove_until`) actually take effect.
    """
    text = state.output.completion if completion is None else completion

    allowed_options = set(answer_character(i) for i in range(len(state.choices)))

    matched = _last_labelled_answer(text, _ANSWER_MARKER_ZH_RE, _PLAIN_LABEL_ZH_RE, allowed_options)

    if matched is None:
        return _fallback_parse_answer(text, allowed_options) or set()

    matched = matched.strip().rstrip('。.')

    if multiple_correct:
        # Handle comma-separated or continuous letters
        matched = matched.replace(' 和 ', '').replace(' ', '').replace('，', ',')
        answers = set(matched.split(',')) if ',' in matched else set(matched)
        return answers if answers.issubset(allowed_options) else set()
    else:
        # Single answer
        return {matched} if matched in allowed_options else set()


def set_choices_based_on_generated_response(state: TaskState, answers: set[str]) -> None:
    true_answers = [answer_index(letter) for letter in answers]

    for i in range(len(state.choices)):
        if i in true_answers:
            state.choices.mark_choice(i, True)
        else:
            state.choices.mark_choice(i, False)


def valid_template(template: str) -> bool:
    """Check if a template has the required capture groups for a multiple choice question"""
    return bool(re.search(r'\{question\}', template) and re.search(r'\{choices\}', template))


class MultipleChoiceTemplate:
    """
    Templates for multiple choice questions.
    """

    SINGLE_ANSWER = SINGLE_ANSWER_TEMPLATE
    SINGLE_ANSWER_COT = SINGLE_ANSWER_TEMPLATE_COT
    MULTIPLE_ANSWER = MULTIPLE_ANSWER_TEMPLATE
    MULTIPLE_ANSWER_COT = MULTIPLE_ANSWER_TEMPLATE_COT
    CHINESE_FEW_SHOT_TEMPLATE = CHINESE_FEW_SHOT_TEMPLATE
    CHINESE_SINGLE_ANSWER_TEMPLATE = CHINESE_SINGLE_ANSWER_TEMPLATE
    CHINESE_SINGLE_ANSWER_TEMPLATE_COT = CHINESE_SINGLE_ANSWER_TEMPLATE_COT
    CHINESE_MULTIPLE_ANSWER_TEMPLATE = CHINESE_MULTIPLE_ANSWER_TEMPLATE
    CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT = CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT


def answer_character(index: int) -> str:
    r"""
    Helper to go from array index to char, for example:

        0 -> 'A', 1 -> 'B', etc
    """
    if index < 26:
        return chr(ord('A') + index)
    else:
        return str(index - 25)


def answer_index(char: str) -> int:
    r"""
    Helper to go from char to array index, for example:

        'A' -> 0, 'B' -> 1, etc
    """
    if char.isalpha() or char == ',' or char == ' ':
        return ord(char.upper()) - ord('A')
    elif char.isnumeric():
        return 25 + int(char)
    else:
        raise ValueError(f'Unepxected multiple choice answer: {char} (must be a letter or number)')
