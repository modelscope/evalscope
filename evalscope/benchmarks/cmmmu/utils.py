import random
import re
from collections import Counter


def eval_cmmmu(entry):
    correct = False
    parsed_pred = entry['parsed_pred']
    if entry.get('type') == '选择':
        if parsed_pred == entry['answer']:
            correct = True

    elif entry.get('type') == '填空':
        norm_answers = normalize_str(entry['answer'], entry['answer'])

        for pred in parsed_pred:
            # already normalized
            if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
                for norm_ans in norm_answers:
                    # only see if the string answer in the string pred
                    # print(norm_ans, pred)
                    if isinstance(norm_ans, str) and norm_ans in pred:
                        if not correct:
                            correct = True
                        break
            else:  # it's a number
                if pred in norm_answers:
                    if not correct:
                        correct = True
                    break

    else:
        positive_keywords = ['正确', '对', '准确', '肯定', '对的']
        negative_keywords = ['不对', '错误', '不正确', '不准确', '不合适', '否定', '错的', '错']
        ambiguous_keywords = ['对错', '是否正确', '否正确', '或者', '是否', '正确性', '对不']

        def judge_similarity(pred_list, positive_keywords, negative_keywords):
            positive_count = 0
            negative_count = 0

            for pred in pred_list:
                if any(pos_word in pred for pos_word in positive_keywords):
                    positive_count += 1
                elif any(neg_word in pred for neg_word in negative_keywords):
                    negative_count += 1

            if positive_count > negative_count:
                return '对'
            elif negative_count > positive_count:
                return '错'
            else:
                return random.choice(['对', '错'])

        answer = entry['answer']
        parsed_pred = [word for word in parsed_pred if not any(ambiguous in word for ambiguous in ambiguous_keywords)]
        result = judge_similarity(parsed_pred, positive_keywords, negative_keywords)
        if result == answer:
            correct = True
    return correct


def get_multi_choice_prediction(response, all_choices, index2ans):
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = ' ' + response + ' '  # add space to avoid partial match

    candidates = []

    for choice in all_choices:  # (A) (B) (C) (D)
        # Add the choice to candidates each time it appears in the response
        candidates.extend([choice for _ in range(response.count(f'({choice})'))])

    if len(candidates) == 0:
        for choice in all_choices:  # A B C D
            # Similarly, add the choice for each occurrence
            candidates.extend([choice for _ in range(response.count(f'{choice}'))])

    if len(candidates) == 0 and len(response.split()) >= 1:
        for index, ans in index2ans.items():
            # Add index for each occurrence of ans in response
            candidates.extend([index for _ in range(response.count(ans))])

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) >= 1:
        for index, ans in index2ans.items():
            if ans in response:
                candidates.append(index)

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        return random.choice(all_choices)
        # return ''
    else:
        # Count the occurrence of each candidate
        candidate_counts = Counter(candidates)

        # Select the most frequent candidates
        max_count = max(candidate_counts.values())
        most_frequent_candidates = [c for c in all_choices if candidate_counts.get(c, 0) == max_count]

        # Combine the most frequent candidates in ABCD order
        return ''.join(most_frequent_candidates)


def extract_numbers(string):
    # Pattern for numbers with Chinese commas
    pattern_commas = r'-?\d{1,3}(?:，\d{3})+'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without Chinese commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+)(?![eE][+-]?\d+)(?!，\d)'

    # Extract numbers with Chinese commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without Chinese commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_is_number(string):
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def count_letters(string):
    return sum(c.isalpha() and 'a' <= c <= 'z' or 'A' <= c <= 'Z' for c in string)


def normalize_str(string, answer):
    # check if characters in the string

    # if number, numerize it.
    if string is None:
        return [string]
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        if len(string) > len(answer) + 20 or count_letters(string) > count_letters(answer) + 2:
            return []
        return [string]


def get_fill_blank_prediction(response, answer):
    """get the prediction from the generated response,
    return a list of predicted strings or numbers"""

    def get_key_subresponses(response):
        key_responses = []
        response = response.strip('。').strip()
        sub_responses = re.split(r'。|\n', response)
        indicators_of_keys = ['是', '为', '所以', '等于', '方案', '选择', '正确答案', '因此', '最后', '答案', '结果']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            # the shortest response that may contain the answer (tail part of the response)
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [':', ',', '.', '!', '?', ';', ':', "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i], answer))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_TF_prediction(response):
    """get the prediction from the generated response,
    return a list of predicted strings or numbers"""

    def get_key_subresponses(response):
        key_responses = []
        response = response.strip('。').strip()
        sub_responses = re.split(r'。|\n', response)
        indicators_of_keys = ['是', '为', '所以', '判断', '陈述', '说法', '表达', '答案', '结果']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [':', ',', '.', '!', '?', ';', ':', "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_multi_choice_info(options):
    start_chr = 'A'
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices
