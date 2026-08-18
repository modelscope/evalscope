import re
import string
from typing import Any

# Chinese punctuation that ``string.punctuation`` does not cover.
ZH_PUNCTUATION = '！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、，；：《》【】'

# SLAKE stores closed-ended answers with several surface forms for the same polarity
# (Chinese targets in the test split include 是的 / 是 / 有 / 包含 / 可以 / 存在 for "yes" and
# 不是 / 否 / 没有 / 不包含 / 不可以 / 不正常 for "no"), so both sides of the comparison are
# collapsed onto a single label before matching.
EN_YES = {'yes', 'yeah', 'yep', 'true'}
EN_NO = {'no', 'nope', 'false'}
ZH_YES = {'是', '是的', '有', '包含', '存在', '可以', '能', '会'}
ZH_NO = {'否', '不是', '没有', '无', '不包含', '不存在', '不可以', '不能', '不会', '不正常'}

_PUNCTUATION_TABLE = str.maketrans({char: ' ' for char in string.punctuation + ZH_PUNCTUATION})

_ANSWER_PATTERN = re.compile(r'ANSWER:\s*(.*)', flags=re.IGNORECASE)

# The official med-vqa answer preprocessing (``preprocess_answer`` in tools/create_label.py) maps
# word-form numbers to digits and drops articles before comparing answers. The Chinese number
# words are added here for the same reason as the X-Ray aliases below: every ``Quantity`` reference
# is an Arabic digit, while the Chinese prompt asks for a Chinese answer.
_NUMBER_WORDS = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
    '零': '0',
    '一': '1',
    '两': '2',
    '二': '2',
    '三': '3',
    '四': '4',
    '五': '5',
    '六': '6',
    '七': '7',
    '八': '8',
    '九': '9',
    '十': '10',
}
_ARTICLES = {'a', 'an', 'the'}

# Modality references are stored in English even in the Chinese half of the dataset, while the
# Chinese prompt asks for a Chinese answer, so the Chinese spellings of X-Ray have to resolve to
# the same label as the reference.
_ALIASES = {
    'x ray': 'xray',
    'xray': 'xray',
    'x光': 'xray',
    'x 光': 'xray',
    'x射线': 'xray',
    'x 射线': 'xray',
}


def normalize_answer(answer: Any) -> str:
    """Normalize a SLAKE answer so that only meaningful differences remain.

    Lower-cases, drops parenthesised asides and punctuation, collapses whitespace, maps word-form
    numbers to digits and drops articles as the official evaluation does, then maps yes/no synonyms
    and the X-Ray spellings onto a canonical form.

    Args:
        answer: Raw answer string (or any value convertible to one).

    Returns:
        The normalized answer used for exact-match scoring.
    """
    text = '' if answer is None else str(answer)
    text = text.strip().lower()
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'（[^）]*）', ' ', text)
    text = text.translate(_PUNCTUATION_TABLE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(word for word in map(_normalize_word, text.split()) if word not in _ARTICLES)
    if text in _ALIASES:
        return _ALIASES[text]
    if text in EN_YES or text in ZH_YES:
        return 'yes'
    if text in EN_NO or text in ZH_NO:
        return 'no'
    return text


def _normalize_word(word: str) -> str:
    """Map one word-form number to its digit, tolerating a trailing Chinese counter word."""
    if word.endswith('个') and (word[:-1].isdigit() or word[:-1] in _NUMBER_WORDS):
        word = word[:-1]
    return _NUMBER_WORDS.get(word, word)


def parse_answer(prediction: str) -> str:
    """Read the answer from the last ``ANSWER:`` line of a reply.

    The prompt itself contains the marker, so a reply that restates the instruction before
    answering must not shadow the real answer: the last occurrence wins. Replies that ignore the
    format are used as-is, which is what makes bare short answers such as ``CT`` score.

    Args:
        prediction: Raw model reply.

    Returns:
        The answer text, or the whole reply when no marker is present.
    """
    matches = _ANSWER_PATTERN.findall(prediction or '')
    if matches:
        return matches[-1].strip().strip('"\'')
    return (prediction or '').strip()
