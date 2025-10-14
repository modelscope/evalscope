# flake8: noqa
import math
import re


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def vqa_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        predict_str = str(predict).lower().strip().replace('\n', ' ')
        for ans in answers:
            answer = str(ans).lower().strip().replace('\n', ' ')
            if len(answer.split()) < 5:
                if answer in predict_str:
                    score = 1
            else:
                dist = levenshtein_distance(predict_str, answer)
                length = max(len(predict_str), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answer = str(answers).lower().strip().replace('\n', ' ')
        predict_str = str(predict).lower().strip().replace('\n', ' ')
        if len(answer.split()) < 5:
            if answer in predict_str:
                score = 1
        else:
            dist = levenshtein_distance(predict_str, answer)
            length = max(len(predict_str), len(answer))
            ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
            ANLS_value = 1 - ANLS_value

            if ANLS_value >= 0.5 and ANLS_value > score:
                score = ANLS_value

    return score


def cn_vqa_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        predict_str = str(predict).lower().strip().replace('\n', ' ').replace(' ', '')
        for ans in answers:
            answer = str(ans).lower().strip().replace('\n', ' ').replace(' ', '')
            if len(answer.split(',')) < 4:
                if answer in predict_str:
                    score = 1
            else:
                dist = levenshtein_distance(predict_str, answer)
                length = max(len(predict_str), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answer = str(answers).lower().strip().replace('\n', ' ').replace(' ', '')
        predict_str = str(predict).lower().strip().replace('\n', ' ').replace(' ', '')
        if len(answer.split(',')) < 4:
            if answer in predict_str:
                score = 1
        else:
            dist = levenshtein_distance(predict_str, answer)
            length = max(len(predict_str), len(answer))
            ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
            ANLS_value = 1 - ANLS_value

            if ANLS_value >= 0.5 and ANLS_value > score:
                score = ANLS_value

    return score


def vqa_evaluation_case_sensitive(predict, answers):
    score = 0
    if isinstance(answers, list):
        predict_str = str(predict).strip().replace('\n', ' ')
        for ans in answers:
            answer = str(ans).strip().replace('\n', ' ')
            if len(answer.split()) < 5:
                if answer in predict_str:
                    score = 1
            else:
                dist = levenshtein_distance(predict_str, answer)
                length = max(len(predict_str), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answer = str(answers).strip().replace('\n', ' ')
        predict_str = str(predict).strip().replace('\n', ' ')
        if len(answer.split()) < 5:
            if answer in predict_str:
                score = 1
        else:
            dist = levenshtein_distance(predict_str, answer)
            length = max(len(predict_str), len(answer))
            ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
            ANLS_value = 1 - ANLS_value

            if ANLS_value >= 0.5 and ANLS_value > score:
                score = ANLS_value

    return score


def extract_first_number(string):
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    return None


def counting_evaluation(predict, answers, eval_method):
    score = 0

    # normalize predict to string for both matching and number extraction
    if isinstance(predict, str):
        predict_str = predict.lower().strip().replace('\n', ' ')
    elif isinstance(predict, (int, float)):
        if isinstance(predict, float) and math.isnan(predict):
            return 0
        predict_str = str(predict).lower().strip().replace('\n', ' ')
    else:
        predict_str = str(predict).lower().strip().replace('\n', ' ')

    if isinstance(answers, list):
        temp_score = 0
        for ans in answers:
            answer = str(ans).lower().strip().replace('\n', ' ')
            if eval_method == 'exact match':
                score = 1 if answer in predict_str else 0
            elif eval_method == 'regression':
                predict_number = extract_first_number(predict_str)
                if predict_number is not None:
                    try:
                        answer_int = int(answer)
                    except ValueError:
                        score = 0
                    else:
                        if predict_number <= 0 or predict_number >= 2 * answer_int:
                            score = 0
                        else:
                            iou = 1 - abs(predict_number - answer_int) / answer_int
                            score = iou if iou > 0.5 else 0
                else:
                    score = 0
            if score > temp_score:
                temp_score = score
        score = temp_score

    else:
        answer = str(answers).lower().strip().replace('\n', ' ')
        if eval_method == 'exact match':
            score = 1 if answer in predict_str else 0
        elif eval_method == 'regression':
            predict_number = extract_first_number(predict_str)
            if predict_number is not None:
                try:
                    answer_int = int(answer)
                except ValueError:
                    score = 0
                else:
                    if predict_number <= 0 or predict_number >= 2 * answer_int:
                        score = 0
                    else:
                        iou = 1 - abs(predict_number - answer_int) / answer_int
                        score = iou if iou > 0.5 else 0
            else:
                score = 0
    return score


def math_expression_evaluation(predict, answers):
    score = 0
    if type(answers) == list:
        for j in range(len(answers)):
            answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
            predict = predict.strip().replace('\n', ' ').replace(' ', '')
            if answer in predict:
                score = 1
    else:
        answers = answers.strip().replace('\n', ' ').replace(' ', '')
        predict = predict.strip().replace('\n', ' ').replace(' ', '')
        if answers in predict:
            score = 1
    return score


def remove_text_tags(latex_str):
    """
    Removes LaTeX \text{...} tags while keeping their content.

    :param latex_str: A string containing LaTeX expressions
    :return: The processed string with \text{...} tags removed
    """

    pattern = r'\\text\{([^{}]*)\}'

    processed_str = re.sub(pattern, r'\1', latex_str)

    return processed_str


def cn_math_expression_evaluation(predict, answers):
    score = 0

    assert len(answers) == 1
    answers = [remove_text_tags(answers[0])]
    predict = remove_text_tags(predict)

    if type(answers) == list:
        for j in range(len(answers)):
            answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
            predict = predict.strip().replace('\n', ' ').replace(' ', '')
            if answer in predict:
                score = 1
    else:
        answers = answers.strip().replace('\n', ' ').replace(' ', '')
        predict = predict.strip().replace('\n', ' ').replace(' ', '')
        if answers in predict:
            score = 1
    return score


if __name__ == '__main__':
    test_predict = 'apple pie and banana'
    test_answers = ['apple', 'banana pie', 'apple pie and orange']

    vqa_score = vqa_evaluation(test_predict, test_answers)
    print(f"VQA evaluation score for predict '{test_predict}' and answers {test_answers}: {vqa_score}")
