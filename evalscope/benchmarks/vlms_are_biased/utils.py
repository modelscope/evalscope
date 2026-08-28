from typing import Any, Dict


def normalize_answer(answer: str) -> str:
    """Normalize an answer using the official lmms-eval comparison rule."""
    return str(answer or '').strip().lower().strip('{}').strip()


def score_answer(prediction: str, ground_truth: str, expected_bias: Any = None) -> Dict[str, float]:
    """Compute the official accuracy and optional bias-ratio indicators."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    normalized_bias = normalize_answer(expected_bias) if expected_bias is not None else None

    is_correct = normalized_prediction == normalized_ground_truth
    matches_bias = normalized_bias is not None and normalized_prediction == normalized_bias

    if not is_correct and not matches_bias:
        prediction_digits = ''.join(character for character in normalized_prediction if character.isdigit())
        ground_truth_digits = ''.join(character for character in normalized_ground_truth if character.isdigit())
        bias_digits = (
            ''.join(character for character in normalized_bias if character.isdigit())
            if normalized_bias is not None
            else ''
        )
        if prediction_digits and ground_truth_digits:
            is_correct = prediction_digits == ground_truth_digits
        if prediction_digits and bias_digits:
            matches_bias = prediction_digits == bias_digits

    result = {'acc': float(is_correct)}
    if normalized_bias is not None:
        result['bias_ratio'] = float(matches_bias)
    return result
