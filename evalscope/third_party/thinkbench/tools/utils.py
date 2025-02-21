import re


def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        last_boxed_content = matches[-1]
        number_pattern = r'-?\d+'
        number_matches = re.findall(number_pattern, last_boxed_content)
        if number_matches:
            return number_matches[-1].strip()
    return None
