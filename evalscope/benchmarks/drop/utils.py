import re
import string

_ARTICLES = re.compile(r'\b(a|an|the)\b', re.UNICODE)


def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_articles(text):
    return _ARTICLES.sub(' ', text)


def _white_space_fix(text):
    return ' '.join(text.split())


def _remove_punc(text):
    exclude = set(string.punctuation)
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in exclude)
    else:
        return text


def _fix_number(text):
    return str(float(text)) if _is_number(text) else text


def _tokenize(text):
    return re.split(' |-', text)


def _normalize(answer):
    tokens = [
        _white_space_fix(_remove_articles(_fix_number(_remove_punc(token.lower())))) for token in _tokenize(answer)
    ]
    tokens = [token for token in tokens if token.strip()]
    normalized = ' '.join(tokens).strip()
    return normalized
