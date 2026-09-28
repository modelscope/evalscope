GENERAL_ORM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
"""  # noqa: E501

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


def extract_answer(response: str) -> str:
    """Extract mathematics upstream; boolean answers follow the dataset's type rule."""
    from evalscope.metrics.math.parser import extract_answer as extract

    boolean = _boolean(response)
    return str(boolean) if boolean is not None else extract(response)


def _boolean(text: str) -> bool | None:
    import re

    match = re.search(
        r'(?:answer\s*(?:is|:)\s*|\A)\(?\s*(true|false|yes|no)\s*\)?[.!]?\s*\Z',
        text.strip(),
        re.IGNORECASE,
    )
    if match:
        return match[1].lower() in ('true', 'yes')
    return None


def get_acc(prediction: str, gt: str, answer_type: str, cot: bool = True) -> int:
    """Preserve explicit relative tolerance (0.15%) and typed boolean comparison.

    ``cot`` is deprecated: extraction always follows Math-Verify.
    """
    import warnings

    from evalscope.metrics.math.contracts import InvalidMathReference
    from evalscope.metrics.math.parser import compare_answers

    if not cot:
        warnings.warn('cot is deprecated; extraction follows Math-Verify', DeprecationWarning, stacklevel=2)
    if answer_type == 'bool':
        gold = _boolean(str(gt))
        if gold is None:
            raise InvalidMathReference(f'Invalid DocMath boolean reference: {gt!r}')
        return int(_boolean(str(prediction)) is gold)
    if answer_type not in ('int', 'float', 'float64'):
        raise InvalidMathReference(f'Unsupported DocMath answer type: {answer_type!r}')
    return int(compare_answers(str(prediction), str(gt), relative_tolerance=0.0015, numeric_reference=True).matched)
