"""Answer processing from the official GT-Vision-Lab/VQA evaluator (vqaEval.py)."""

import re
import string

# https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
_CONTRACTIONS = {
    'aint': "ain't",
    'arent': "aren't",
    'cant': "can't",
    'couldve': "could've",
    'couldnt': "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    'didnt': "didn't",
    'doesnt': "doesn't",
    'dont': "don't",
    'hadnt': "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    'hasnt': "hasn't",
    'havent': "haven't",
    'hed': "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    'hes': "he's",
    'howd': "how'd",
    'howll': "how'll",
    'hows': "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    'Im': "I'm",
    'Ive': "I've",
    'isnt': "isn't",
    'itd': "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    'itll': "it'll",
    "let's": "let's",
    'maam': "ma'am",
    'mightnt': "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    'mightve': "might've",
    'mustnt': "mustn't",
    'mustve': "must've",
    'neednt': "needn't",
    'notve': "not've",
    'oclock': "o'clock",
    'oughtnt': "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    'shant': "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    'shouldve': "should've",
    'shouldnt': "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": 'somebodyd',
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    'somebodyll': "somebody'll",
    'somebodys': "somebody's",
    'someoned': "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    'someonell': "someone'll",
    'someones': "someone's",
    'somethingd': "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    'somethingll': "something'll",
    'thats': "that's",
    'thered': "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    'therere': "there're",
    'theres': "there's",
    'theyd': "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    'theyll': "they'll",
    'theyre': "they're",
    'theyve': "they've",
    'twas': "'twas",
    'wasnt': "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    'weve': "we've",
    'werent': "weren't",
    'whatll': "what'll",
    'whatre': "what're",
    'whats': "what's",
    'whatve': "what've",
    'whens': "when's",
    'whered': "where'd",
    'wheres': "where's",
    'whereve': "where've",
    'whod': "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    'wholl': "who'll",
    'whos': "who's",
    'whove': "who've",
    'whyll': "why'll",
    'whyre': "why're",
    'whys': "why's",
    'wont': "won't",
    'wouldve': "would've",
    'wouldnt': "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    'yall': "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    'youd': "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    'youll': "you'll",
    'youre': "you're",
    'youve': "you've",
}
_PUNCTUATION = (
    ';',
    '/',
    '[',
    ']',
    '"',
    '{',
    '}',
    '(',
    ')',
    '=',
    '+',
    '\\',
    '_',
    '-',
    '>',
    '<',
    '@',
    '`',
    ',',
    '?',
    '!',
)
_DIGIT_COMMA = re.compile(r'\d,\d')
_PERIOD = re.compile(r'\.(?!\d)')
_ARTICLES = {'a', 'an', 'the'}
_NUMBER_MAP = {
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
}


def normalize_annotated_answer(answer: str) -> str:
    """Normalize a prediction or reference for a question with differing human answers."""
    text = answer
    has_numeric_comma = _DIGIT_COMMA.search(answer) is not None
    for mark in _PUNCTUATION:
        remove = has_numeric_comma or mark + ' ' in answer or ' ' + mark in answer
        text = text.replace(mark, '' if remove else ' ')
    # The reference passes re.UNICODE as the third (count) argument of Pattern.sub.
    text = _PERIOD.sub('', text, count=32)
    words = (_NUMBER_MAP.get(word, word) for word in text.lower().split())
    return ' '.join(_CONTRACTIONS.get(word, word) for word in words if word not in _ARTICLES)


def normalize_exact_answer(answer: str) -> str:
    """Keep the legacy normalized exact-match and single-reference comparison."""
    text = answer.lower().strip().replace('\n', ' ').replace('\t', ' ')
    punctuation = string.punctuation.replace(':', '')
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = ''.join(' ' if c in punctuation else c for c in text)
    words = []
    for word in text.split():
        mapped = _NUMBER_MAP.get(word, word)
        if mapped not in _ARTICLES:
            words.append(mapped)
    return ' '.join(words)
