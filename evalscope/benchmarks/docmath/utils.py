import math
import numpy as np
import re
from sympy import Rational

from evalscope.utils.logger import get_logger

logger = get_logger()

GENERAL_ORM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""  # noqa: E501

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


def round_up_to_decimal(number, decimals):
    factor = 10**decimals
    return math.ceil(number * factor) / factor


def is_number(string):
    pattern = r'^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$'
    match = re.match(pattern, string)
    return bool(match)


def is_scientific_number(string):
    pattern = r'^[-+]?\d+(\.\d+)?e[-]?\d+$'
    match = re.match(pattern, string)
    return bool(match)


def normalize(prediction: str):
    # Preprocessing the string [Stage 1]
    prediction = prediction.strip()
    prediction = prediction.rstrip('.')
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else '0'

    for money in ['£', '€', '¥', 'million', 'billion', 'thousand', 'US', 'USD', 'RMB']:
        prediction = prediction.replace(money, '')

    # Replace special tokens
    if '=' in prediction:
        prediction = prediction.split('=')[-1].strip()
    if '≈' in prediction:
        prediction = prediction.split('≈')[-1].strip()
    if '`' in prediction:
        prediction = prediction.replace('`', '')
    if '%' in prediction:
        prediction = prediction.replace('%', '')
    if '$' in prediction:
        prediction = prediction.replace('$', '')
    if '°' in prediction:
        prediction = prediction.replace('°', '')

    # Detect the boolean keyword in the generation
    if prediction in ['true', 'yes', 'false', 'no']:
        if prediction == 'true' or prediction == 'yes':
            prediction = 'True'
        else:
            prediction = 'False'
    if 'True' in prediction or 'False' in prediction:
        prediction = 'True' if 'True' in prediction else 'False'

    # Detect the approximation keyword
    if 'approximately' in prediction:
        prediction = prediction.replace('approximately', '').strip()
    if ' or ' in prediction:
        prediction = prediction.split(' or ')[0]

    # Drop the units before and after the number
    if re.match(r'[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$', prediction).group(1)
    if re.match(r'[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$', prediction).group(1)
    if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction).group(1)
    if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction).group(1)

    # Preprocessing the number [Stage 1]
    if '10^' in prediction:
        prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)
    if ' x ' in prediction:
        prediction = prediction.replace(' x ', '*')
    if ' × ' in prediction:
        prediction = prediction.replace(' × ', '*')
    if is_number(prediction):
        prediction = prediction.replace(',', '')

    # Preprocessing the option [Stage 3]
    if '(a)' in prediction or '(b)' in prediction or '(c)' in prediction or '(d)' in prediction:
        prediction = '"' + re.search(r'\([a-d]\)', prediction).group(0) + '"'

    # If the prediction is empty, use dummy '0'
    if not prediction:
        prediction = '0'

    # Converting the string answer to a number/list/bool/option
    try:
        prediction = eval(prediction)
    except Exception:
        # TO CHECK
        prediction = 0

    # Performing common type conversion
    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if isinstance(prediction[0], complex):
            prediction = [tmp.real for tmp in prediction]
        elif isinstance(prediction[0], Rational):
            prediction = [float(tmp) for tmp in prediction]
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real
        elif isinstance(prediction, Rational):
            prediction = float(prediction)

    return prediction


def extract_answer(response: str):
    """Parses the final answer from the model's response text.

    Args:
        response: Text extracted from the model's response

    Returns:
        The final answer as a numeric value (string), or None if not found
    """
    # Remove any asterisks or other unwanted characters
    response = response.replace('*', '')
    response = response.replace('(', '')
    response = response.replace(')', '')

    # Search for the pattern 'the answer is {final answer}.'
    match = re.search(r'the answer is (\=?\≈?\`?\%?\$?\°?\£?\€?\¥?-?[0-9\.,]+)', response, re.IGNORECASE)

    if match:
        # Remove commas from the matched number (if any)
        res = match.group(1).replace(',', '').rstrip('.')
        return res
    else:
        return response


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.0015
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        raise ValueError(p)

    v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
    if (v1 != 0 and v2 != 0) and int(math.log10(v1 / v2)) == math.log10(v1 / v2):
        return True

    if v2 <= v1 / 50 and within_eps(pred=v2 * 100, gt=v1):
        return True
    elif v2 <= v1 / 500 and within_eps(pred=v2 * 1000, gt=v1):
        return True
    elif v2 <= v1 / 50000 and within_eps(pred=v2 * 100000, gt=v1):
        return True

    if round_up_to_decimal(v1, 2) == round_up_to_decimal(v2, 2):
        return True

    return within_eps(pred=p, gt=gt)


def get_acc(prediction, gt, cot=True):
    try:
        if cot:
            prediction = normalize(prediction)
        else:
            prediction = float(prediction)

        answer_type = type(gt).__name__
        assert answer_type in ['int', 'float', 'float64', 'bool'], answer_type
        if isinstance(prediction, (str, int, float, bool)) or isinstance(prediction, list):
            # Comparing prediction against the reference
            if answer_type in ['bool']:
                acc = int(prediction == gt)
            elif answer_type == 'int':
                acc = int(compare_two_numbers(prediction, gt))
            elif answer_type == 'float' or answer_type == 'float64':
                acc = int(compare_two_numbers(prediction, gt))
            else:
                acc = 0
        else:
            acc = 0
            logger.error('Error: ', prediction, type(prediction))
        return acc
    except Exception:
        return 0
