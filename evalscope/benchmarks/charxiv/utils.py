# flake8: noqa: E501
"""CharXiv question templates and grading utilities.

Based on the official CharXiv evaluation:
https://github.com/princeton-nlp/CharXiv
"""

from typing import Dict, Optional

# Descriptive question response templates (indexed 1-19)
# The {} placeholder is filled with a prefix like "For the current plot, " or "For {subplot_loc}, "
DESCRIPTIVE_RESP_INST: Dict[int, str] = {
    1: (
        '{}what is its title?\n'
        '* Your final answer should be the most relevant title of the plot that is explicitly written.\n'
        '* If the plot does not have an explicit title or contains only a letter, answer \'Not Applicable\'.'
    ),
    2: (
        '{}what is the label of the x-axis?\n'
        '* Your final answer should be the label of the x-axis that is explicitly written, including the case when '
        'x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the '
        'plot, answer the label of the x-axis at the bottom.\n'
        '* If the plot does not have an explicit x-axis label, answer \'Not Applicable\'.'
    ),
    3: (
        '{}what is the label of the y-axis?\n'
        '* Your final answer should be the label of the y-axis that is explicitly written, including the case when '
        'y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the '
        'plot, answer the label of the y-axis at the left.\n'
        '* If the plot does not have an explicit y-axis label, answer \'Not Applicable\'.'
    ),
    4: (
        '{}what is the leftmost labeled tick on the x-axis?\n'
        '* Your final answer should be the tick value on the x-axis that is explicitly written. '
        'Ignore units or scales that are written separately from the tick.'
    ),
    5: (
        '{}what is the rightmost labeled tick on the x-axis?\n'
        '* Your final answer should be the tick value on the x-axis that is explicitly written. '
        'Ignore units or scales that are written separately from the tick.'
    ),
    6: (
        '{}what is the spatially lowest labeled tick on the y-axis?\n'
        '* Your final answer should be the tick value on the y-axis that is explicitly written. '
        'Ignore units or scales that are written separately from the tick.'
    ),
    7: (
        '{}what is the spatially highest labeled tick on the y-axis?\n'
        '* Your final answer should be the tick value on the y-axis that is explicitly written. '
        'Ignore units or scales that are written separately from the tick.'
    ),
    8: (
        '{}what is difference between consecutive numerical tick values on the x-axis?\n'
        '* If the plot does not have an explicit x-axis tick value, or if the tick values are not numerical, '
        'or if the difference is not constant between all consecutive tick values, answer "Not Applicable".'
    ),
    9: (
        '{}what is difference between consecutive numerical tick values on the y-axis?\n'
        '* If the plot does not have an explicit y-axis tick value, or if the tick values are not numerical, '
        'or if the difference is not constant between all consecutive tick values, answer "Not Applicable".'
    ),
    10: (
        '{}how many lines are there?\n'
        '* Your final answer should be the number of lines in the plot. Ignore grid lines, tick marks, '
        'and any vertical or horizontal auxiliary lines.\n'
        '* If the plot does not contain any lines or is not considered a line plot, answer "Not Applicable".'
    ),
    11: (
        '{}do any lines intersect?\n'
        '* Your final answer should be "Yes" if any lines intersect, and "No" otherwise.\n'
        '* If the plot does not contain any lines or is not considered a line plot, answer "Not Applicable".'
    ),
    12: (
        '{}how many discrete labels are there in the legend?\n'
        '* Your final answer should account for only labels relevant to the plot in the legend.\n'
        '* If the plot does not have a legend, answer "Not Applicable".'
    ),
    13: (
        '{}what are the names of the labels in the legend?\n'
        '* You should write down the labels from top to bottom, then from left to right and separate '
        'the labels with commas.\n'
        '* If the plot does not have a legend, answer "Not Applicable".'
    ),
    14: (
        '{}what is the difference between the maximum and minimum values of the tick labels on the '
        'continuous legend (i.e., colorbar)?\n'
        '* You should remove the percentage sign (if any) in your answer.\n'
        '* If the plot does not have an explicit colorbar-based continuous legend, answer "Not Applicable".'
    ),
    15: (
        '{}what is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?\n'
        '* You should remove the percentage sign (if any) in your answer.\n'
        '* If the plot does not have an explicit colorbar-based continuous legend, answer "Not Applicable".'
    ),
    16: (
        '{}what is the general trend of data from left to right?\n'
        '* Your final answer should be within a few words, such as "increases", "increases then stabilizes".'
    ),
    17: (
        '{}What is the total number of explicitly labeled ticks across all axes?\n'
        '* Your final answer should be the total number of explicitly labeled ticks across all axes.'
    ),
    18: (
        'What is the layout of the subplots?\n'
        '* Your final answer should follow "n by m" format, where n is the number of rows and m is '
        'the number of columns.\n'
        '* If the plot does not contain subplots, answer "1 by 1".'
    ),
    19: (
        'What is the number of subplots?\n'
        '* Your final answer should be the total number of subplots in the plot.\n'
        '* If the plot does not contain subplots, answer "1".'
    ),
}

# Reasoning question response instructions (indexed by reasoning_a_type 1-4)
REASONING_RESP_INST: Dict[int, str] = {
    1: (
        '{}\n'
        '* Your final answer must be grounded to some text that is explicitly written and relevant '
        'to the question in the chart.\n'
        '* If you need to answer multiple terms, separate them with commas.\n'
        '* Unless specified in the question (such as answering with a letter), you are required to '
        'answer the full names of subplots and/or labels by default.'
    ),
    2: (
        '{}\n'
        '* If there are options in the question, your final answer must conform to one of the options.\n'
        '* If there are additional instructions in the question, follow them accordingly.\n'
        '* If there are neither options nor additional instructions, you are allowed to respond with '
        'a short phrase only.'
    ),
    3: (
        '{}\n'
        '* Your final answer must be grounded to a number that is explicitly written and relevant to '
        'the question in the chart, even if it\'s an approximate value.\n'
        '* You are allowed to extract numbers within some text when needed.'
    ),
    4: ('{}\n'
        '{}'),
}


def get_descriptive_question_text(q_id: int, subplot_loc: Optional[str] = None) -> str:
    """Generate the full descriptive question text from a template ID.

    Follows the official CharXiv prompt construction:
    - qid 18/19: no prefix (about overall layout)
    - subplot_loc is a string: "For {subplot_loc}, "
    - subplot_loc is None: "For the current plot, "

    Args:
        q_id: Question template ID (1-19)
        subplot_loc: Subplot location string or None

    Returns:
        The complete question text string
    """
    template = DESCRIPTIVE_RESP_INST.get(q_id, '')
    if not template:
        return ''

    # Questions 18 and 19 don't use a prefix (they ask about subplot layout itself)
    if q_id in (18, 19):
        return template

    # Construct prefix following official logic
    if subplot_loc and isinstance(subplot_loc, str):
        prefix = f'For {subplot_loc}, '
    else:
        prefix = 'For the current plot, '

    return template.format(prefix)


def _get_number_instruction(answer: str) -> str:
    """Generate number format instruction based on the answer's decimal places.

    Following official CharXiv: specifies whether the answer is an integer or
    how many decimal places it has.
    """
    parts = str(answer).split('.')
    if len(parts) == 1:
        return '* Your final answer must be an exact integer.'
    else:
        num_decimal = len(parts[1])
        return f'* Your final answer must be a number with {num_decimal} decimal places.'


def get_reasoning_question_text(question: str, reasoning_a_type: int, answer: str = '') -> str:
    """Generate the full reasoning question text with instructions.

    Args:
        question: The reasoning question text
        reasoning_a_type: Answer type (1-4)
        answer: The reference answer (needed for type 4 to determine decimal format)

    Returns:
        The complete question text with instructions
    """
    template = REASONING_RESP_INST.get(reasoning_a_type, REASONING_RESP_INST[1])
    if reasoning_a_type == 4:
        # Type 4 has two placeholders: question + number format instruction
        number_inst = _get_number_instruction(answer)
        return template.format(question, number_inst)
    return template.format(question)


# =============================================================================
# Grading prompts (following official CharXiv evaluation protocol)
# =============================================================================

# Mapping from descriptive question ID to rubric category
DESCRIPTIVE_QID_TO_RUBRIC: Dict[int, str] = {
    1: 'title',
    2: 'ocr',
    3: 'ocr',
    4: 'ocr',
    5: 'ocr',
    6: 'ocr',
    7: 'ocr',
    8: 'quant',
    9: 'quant',
    10: 'quant',
    12: 'quant',
    14: 'quant',
    15: 'quant',
    17: 'quant',
    19: 'quant',
    11: 'bool',
    13: 'enum',
    16: 'trend',
    18: 'layout',
}

# Overarching question text for each descriptive qid (used in grading prompt)
DESCRIPTIVE_GRADING_QMAP: Dict[int, str] = {
    1: 'What is the title of the plot?',
    2: 'What is the label of the x-axis?',
    3: 'What is the label of the y-axis?',
    4: 'What is the leftmost labeled tick on the x-axis?',
    5: 'What is the rightmost labeled tick on the x-axis?',
    6: 'What is the spatially lowest labeled tick on the y-axis?',
    7: 'What is the spatially highest labeled tick on the y-axis?',
    8: 'What is difference between consecutive numerical tick values on the x-axis?',
    9: 'What is difference between consecutive numerical tick values on the y-axis?',
    10: 'How many lines are there?',
    11: 'Do any lines intersect?',
    12: 'How many discrete labels are there in the legend?',
    13: 'What are the names of the labels in the legend? (from top to bottom, then left to right)',
    14: 'What is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?',
    15: 'What is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?',
    16: 'What is the general trend of data from left to right?',
    17: 'What is the total number of explicitly labeled ticks across all axes?',
    18: 'What is the layout of the subplots?',
    19: 'What is the number of subplots?',
}

# Rubric + ICL examples per category (from official constants.py)
DESCRIPTIVE_GRADING_ICL: Dict[str, str] = {
    'title': (
        'Rubric:\n'
        '    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. '
        "It's acceptable to have different grammar or form (e.g., \u03b1 and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). "
        "It's acceptable to omit letter prefixes (e.g., (a) Increment over time and Increment over time).\n"
        '    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer.\n'
        '    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.\n\n'
        '    ### Example Start ###\n'
        '    T1:\n'
        '    Response 1: The title of the plot is "The number of students in each grade".\n'
        '    Ground Truth 1: The variance of students in each grade\n\n'
        '    T2:\n'
        '    Response 2: There is no title.\n'
        '    Ground Truth 2: Not Applicable\n\n'
        '    T3:\n'
        '    Response 3: A_v^t\n'
        '    Ground Truth 3: A^t_v\n\n'
        '    {\n'
        '        "extract_answer_T1": "The number of students in each grade",\n'
        '        "score_T1": 0\n'
        '        "extract_answer_T2": "Not Applicable",\n'
        '        "score_T2": 1\n'
        '        "extract_answer_T3": "A_v^t",\n'
        '        "score_T3": 1\n'
        '    }\n'
        '    ### Example End ###'
    ),
    'ocr': (
        'Rubric:\n'
        '    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. '
        "It's acceptable to have equivalent grammar or form (e.g., \u03b1 and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). "
        'If the ground truth is a number, the extracted answer should be the number with the exact same value.\n'
        '    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer, '
        'or if the extracted number is different in value from the ground truth number.\n'
        '    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.\n\n'
        '    ### Example Start ###\n'
        '    T1:\n'
        '    Response 1: The answer is 1.0\n'
        '    Ground Truth 1: 1.00\n\n'
        '    T2:\n'
        '    Response 2: By manually inspecting the plot, the final answer should be 0.\n'
        '    Ground Truth 2: Not Applicable\n\n'
        '    T3:\n'
        '    Response 3: A_v^t\n'
        '    Ground Truth 3: A^t_v\n\n'
        '    {\n'
        '        "extract_answer_T1": 1.0,\n'
        '        "score_T1": 1\n'
        '        "extract_answer_T2": 0,\n'
        '        "score_T2": 0\n'
        '        "extract_answer_T3": "A_v^t",\n'
        '        "score_T3": 1\n'
        '    }\n'
        '    ### Example End ###'
    ),
    'quant': (
        'Rubric:\n'
        '    * Give a score of 1 if and only if the extracted answer and the ground truth answer are numbers with the exact same value.\n'
        '    * Give a score of 0 if the extracted answer is different in value from the ground truth answer.\n'
        '    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.\n\n'
        '    ### Example Start ###\n'
        '    T1:\n'
        '    Response 1: 5\n'
        '    Ground Truth 1: 6\n\n'
        '    T2:\n'
        '    Response 2: 0\n'
        '    Ground Truth 2: Not Applicable\n\n'
        '    T3:\n'
        '    Response 3: 4\n'
        '    Ground Truth 3: 4\n\n'
        '    {\n'
        '        "extract_answer_T1": 5,\n'
        '        "score_T1": 0\n'
        '        "extract_answer_T2": 0,\n'
        '        "score_T2": 0\n'
        '        "extract_answer_T3": 4,\n'
        '        "score_T3": 1\n'
        '    }\n'
        '    ### Example End ###'
    ),
    'bool': (
        'Rubric:\n'
        '    * Give a score of 1 if and only if the extracted answer and the ground truth answer are the same.\n'
        '    * Give a score of 0 if the extracted answer and the ground truth answer are different.\n'
        '    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.\n\n'
        '    ### Example Start ###\n'
        '    T1:\n'
        '    Response 1: No, there are no intersections.\n'
        '    Ground Truth 1: no\n\n'
        '    T2:\n'
        '    Response 2: No, all the lines are parallel.\n'
        '    Ground Truth 2: Yes\n\n'
        '    T3:\n'
        '    Response 3: There are no lines in the plot.\n'
        '    Ground Truth 3: Not Applicable\n\n'
        '    {\n'
        '        "extract_answer_T1": "No",\n'
        '        "score_T1": 1\n'
        '        "extract_answer_T2": "No",\n'
        '        "score_T2": 0\n'
        '        "extract_answer_T3": "Not Applicable",\n'
        '        "score_T3": 1\n'
        '    }\n'
        '    ### Example End ###'
    ),
    'enum': (
        'Rubric:\n'
        '    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. '
        "It's acceptable to have equivalent grammar or form (e.g., \u03b1 and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). "
        'The order of the terms must be the same.\n'
        '    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer, '
        'or if the order of the terms is different.\n'
        '    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.\n\n'
        '    ### Example Start ###\n'
        '    T1:\n'
        '    Response 1: Here are the names of the labels: A, B, C\n'
        '    Ground Truth 1: B, A, C\n\n'
        '    T2:\n'
        '    Response 2: The labels are T56, B33.\n'
        '    Ground Truth 2: T56,B33,A12\n\n'
        '    T3:\n'
        '    Response 3: \\alpha, \\beta, \\gamma^t_v\n'
        '    Ground Truth 3: \u03b1, \u03b2, \u03b3_v^t\n\n'
        '    {\n'
        '        "extract_answer_T1": "A, B, C",\n'
        '        "score_T1": 0\n'
        '        "extract_answer_T2": "T56, B33",\n'
        '        "score_T2": 0\n'
        '        "extract_answer_T3": "\\\\alpha, \\\\beta, \\\\gamma^t_v",\n'
        '        "score_T3": 1\n'
        '    }\n'
        '    ### Example End ###'
    ),
    'trend': (
        'Rubric:\n'
        '    * Give a score of 1 if and only if the extracted answer and the ground truth answer share the same general trend.\n'
        '    * Give a score of 0 if the extracted answer and the ground truth answer are different in trend expression.\n\n'
        '    ### Example Start ###\n'
        '    T1:\n'
        '    Response 1: there is an increase in the data from left to right\n'
        '    Ground Truth 1: Decreases\n\n'
        '    T2:\n'
        '    Response 2: the curves move up and stay constant\n'
        '    Ground Truth 2: Increases then stabilizes\n\n'
        '    T3:\n'
        '    Response 3: Decreases\n'
        '    Ground Truth 3: Decreases then increases\n\n'
        '    {\n'
        '        "extract_answer_T1": "Increases",\n'
        '        "score_T1": 0\n'
        '        "extract_answer_T2": "Move up and stay constant",\n'
        '        "score_T2": 1\n'
        '        "extract_answer_T3": "Decreases",\n'
        '        "score_T3": 0\n'
        '    }\n'
        '    ### Example End ###'
    ),
    'layout': (
        'Rubric:\n'
        '    * Give a score of 1 if and only if the extracted answer and the ground truth answer are the same '
        'in terms of the number of rows and columns (e.g., n by m).\n'
        '    * Give a score of 0 if the extracted answer is different from the ground truth answer.\n\n'
        '    ### Example Start ###\n'
        '    T1:\n'
        '    Response 1: 2 by 3\n'
        '    Ground Truth 1: 3 by 2\n\n'
        '    T2:\n'
        '    Response 2: the layout is 1 by 1\n'
        '    Ground Truth 2: 1 by 1\n\n'
        '    T3:\n'
        '    Response 3: there are two rows and three columns\n'
        '    Ground Truth 3: 2 by 3\n\n'
        '    {\n'
        '        "extract_answer_T1": "2 by 3",\n'
        '        "score_T1": 0\n'
        '        "extract_answer_T2": "1 by 1",\n'
        '        "score_T2": 1\n'
        '        "extract_answer_T3": "2 by 3",\n'
        '        "score_T3": 1\n'
        '    }\n'
        '    ### Example End ###'
    ),
}

# Reasoning grading instructions per type (from official constants.py)
REASONING_GRADING_INST: Dict[int, str] = {
    1: (
        '### Rules ###\n'
        '* Give a score of 1 if and only if the final answer and the ground truth answer are referring to the same term. '
        "It's acceptable to have different grammar or form (e.g., \u03b1 and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). "
        "It's also acceptable to have different orders of the terms when question asks for multiple terms.\n"
        '* Give a score of 0 if any term (e.g., ACC+ and ACC; P-101 and P=101) is different between the final answer and the ground truth.\n\n'
        '### Your Turn ###\n'
        '* Question: {question}\n'
        '* Ground Truth: {ground_truth}\n'
        '* Response: {response}'
    ),
    2: (
        '### Rules ###\n'
        '* If there are predefined options in the question:\n'
        '    * Give a score of 1 if the final answer matches the ground truth answer exactly.\n'
        '    * Give a score of 0 if the final answer does not match the ground truth answer.\n'
        '* If there are no predefined options in the question:\n'
        '    * Give a score of 1 if the final answer shares the same semantic meaning with the ground truth answer.\n'
        '    * Give a score of 0 if the final answer shares different semantic meanings from the ground truth answer.\n\n'
        '### Your Turn ###\n'
        '* Question: {question}\n'
        '* Ground Truth: {ground_truth}\n'
        '* Response: {response}'
    ),
    3: (
        '### Rules ###\n'
        "* Give a score of 1 if and only if the two numbers are exactly equal in values. It's acceptable to have different notations "
        '(e.g., 0.01 and 10^-2; 1500 and 1.5e3).\n'
        '* Give a score of 0 if the two numbers are different in values.\n\n'
        '### Your Turn ###\n'
        '* Question: {question}\n'
        '* Ground Truth: {ground_truth}\n'
        '* Response: {response}'
    ),
    4: (
        '### Rules ###\n'
        "* Give a score of 1 if and only if the two numbers are exactly equal in values. It's acceptable to have different notations "
        '(e.g., 0.01 and 10^-2; 1500 and 1.5e3).\n'
        '* Give a score of 0 if the two numbers are different in values.\n\n'
        '### Your Turn ###\n'
        '* Question: {question}\n'
        '* Ground Truth: {ground_truth}\n'
        '* Response: {response}'
    ),
}


def build_descriptive_judge_prompt(q_id: int, response: str, ground_truth: str) -> str:
    """Build LLM judge prompt for a descriptive question following official protocol.

    Args:
        q_id: Descriptive question template ID (1-19)
        response: Model's response
        ground_truth: Ground truth answer

    Returns:
        The complete judge prompt string
    """
    rubric_key = DESCRIPTIVE_QID_TO_RUBRIC.get(q_id, 'ocr')
    question = DESCRIPTIVE_GRADING_QMAP.get(q_id, '')
    rubric = DESCRIPTIVE_GRADING_ICL[rubric_key]

    prompt = (
        'You will be given 1 pair of ground truth answer and model response under an overarching question. '
        'You need to extract the final answer from the model response, compare it with the ground truth answer, '
        'and then assign a binary score. Avoid providing explanations in your response. '
        'If there is no provided model response, please leave the extracted answer empty and give a score of 0. '
        'Your response must follow json format with keys ["extract_answer_T1", "score_T1"] '
        'where the value for "extract_answer_T1" is your extracted answer and "score_T1" is an integer in [0, 1] '
        'based on the following rules:\n\n'
        f'Overarching Question: {question}\n\n'
        f'{rubric}\n\n'
        f'T1:\nResponse 1: {response}\nGround Truth 1: {ground_truth}\n'
    )
    return prompt


def build_reasoning_judge_prompt(reasoning_a_type: int, question: str, ground_truth: str, response: str) -> str:
    """Build LLM judge prompt for a reasoning question following official protocol.

    Args:
        reasoning_a_type: Reasoning answer type (1-4)
        question: The original question text
        ground_truth: Ground truth answer
        response: Model's response

    Returns:
        The complete judge prompt string
    """
    prefix = (
        'You will be given a question, a ground truth answer and a model response. '
        'You need to extract the final answer from the model response, compare it with the ground truth answer, '
        'and then assign a binary score. Avoid providing explanations in your response. '
        'If there is no provided model response, please leave the extracted answer empty and give a score of 0.\n\n'
        'Your response must follow json format with keys ["extract_answer", "score"] '
        'where the value of the score is an integer in [0, 1]. You must follow the scoring rules:\n'
    )

    inst_template = REASONING_GRADING_INST.get(reasoning_a_type, REASONING_GRADING_INST[1])
    inst = inst_template.replace('{question}', question).replace('{ground_truth}',
                                                                 ground_truth).replace('{response}', response)

    return prefix + inst
