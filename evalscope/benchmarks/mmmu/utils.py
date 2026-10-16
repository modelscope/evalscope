"""Open-question scoring from the official MMMU evaluation (MMMU-Benchmark/MMMU, mmmu/utils/eval_utils.py)."""

import re


def check_is_number(string):
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        return False


def normalize_str(string):
    """Lower-case a string answer; turn a numeric one into a float rounded to 2 decimals."""
    string = string.strip()
    if check_is_number(string):
        return [round(float(string.replace(',', '')), 2)]
    string = string.lower()
    if len(string) == 1:
        return [' ' + string, string + ' ']  # avoid trivial matches
    return [string]


def extract_numbers(string):
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'
    return (
        re.findall(pattern_commas, string) + re.findall(pattern_scientific, string) + re.findall(pattern_simple, string)
    )


def parse_open_response(response):
    """Return the normalized candidate answers (strings and numbers) found in a response."""

    def get_key_subresponses(response):
        response = response.strip().strip('.').lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ', 'thus ', 'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # the last sub-response may be just an equation
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    tail = resp.split(indicator)[-1].strip()
                    if not shortest_key_response or len(tail) < len(shortest_key_response):
                        shortest_key_response = tail
            if shortest_key_response and shortest_key_response.strip() not in [':', ',', '.', '!', '?', ';', "'"]:
                key_responses.append(shortest_key_response)
        return key_responses or [response]

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))
    normalized = []
    for pred in pred_list:
        normalized.extend(normalize_str(pred))
    return list(set(normalized))


def eval_open(gold_i, pred_i):
    """A list gold accepts any of its answers; strings match by inclusion, numbers by equality."""
    norm_answers = []
    for answer in gold_i if isinstance(gold_i, list) else [gold_i]:
        norm_answers.extend(normalize_str(answer))
    for pred in pred_i:
        if isinstance(pred, str):
            if any(isinstance(norm_ans, str) and norm_ans in pred for norm_ans in norm_answers):
                return True
        elif pred in norm_answers:
            return True
    return False
