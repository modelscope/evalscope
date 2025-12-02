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


def _fallback_parse_answer(completion: str) -> Optional[set[str]]:
    # Fallback to find the last upper case letter
    for letter in reversed(completion):
        if letter.isupper():
            return {letter}
    return None


def parse_answers(state: TaskState, multiple_correct: bool = False) -> set[str]:
    """
    Convenience function for extracting answers from the state output.

    The generated response must be in the format 'ANSWER: <answers>',
    otherwise we can't extract what the model thinks is "true". We can be a
    bit flexible whether these are "AB" vs "A,B" vs "A B".

    However, if the answer isn't in the expected format the model has
    failed in the task so we'll ultimately just mark it as incorrect
    """
    # First check whether the string strictly ends with the expected answer
    # In this case, we're looking for a single line which contains the expected
    # ANSWER: <answer> string with only whitespace or a period/full stop at the end.
    match = re.search(
        r'(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)',
        state.output.completion,
        flags=re.MULTILINE,
    )

    # If we couldn't match the strict version, we can try the less strict
    # version for backward compatibility
    if match is None:
        match = re.search(
            r'(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)',
            state.output.completion,
        )

    if match is None:
        fallback_answer = _fallback_parse_answer(state.output.completion)
        if fallback_answer:
            return fallback_answer

    if match is None:
        return set()

    matched = match.group(1)

    # Strip trailing period / full stop
    matched = matched.strip()
    matched = matched.rstrip('.')

    allowed_options = set(answer_character(i) for i in range(len(state.choices)))

    if multiple_correct:
        # Match must contain only the allowed choices
        # (may be separated by commas, spaces, the word 'and', or nothing at all)

        matched = matched.replace(' and ', '')

        matched = matched.replace(' ', '')

        split_comma = set(matched.split(','))
        if split_comma.issubset(allowed_options):
            answers = split_comma
            return answers

        split_nothing = set(matched)
        if split_nothing.issubset(allowed_options):
            answers = split_nothing
            return answers

    else:
        # Match must contain a single letter in the allowed choices
        if matched in allowed_options:
            answers = {matched}
            return answers

    return set()


def parse_answers_zh(state: TaskState, multiple_correct: bool = False) -> set[str]:
    """
    Convenience function for extracting answers from the state output in Chinese format.

    The generated response must be in the format '答案：选项',
    otherwise we can't extract what the model thinks is "true". We can be a
    bit flexible whether these are "AB" vs "A,B" vs "A B".
    """
    # Simple pattern to capture answers with optional bold markdown
    pattern = r'答案\s*[:：]\s*([A-Za-z0-9,，]+)'
    match = re.search(pattern, state.output.completion, flags=re.MULTILINE)

    if match is None:
        fallback_answer = _fallback_parse_answer(state.output.completion)
        if fallback_answer:
            return fallback_answer

    if match is None:
        return set()

    matched = match.group(1).strip().rstrip('。.')
    allowed_options = set(answer_character(i) for i in range(len(state.choices)))

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
