"""Shared HMMT answer extraction delegates to Math-Verify."""


def extract_hmmt_answer(prediction: str) -> str:
    """Extract the final mathematical answer with the shared upstream strategy."""
    from evalscope.metrics.math.parser import extract_answer

    return extract_answer(prediction)
