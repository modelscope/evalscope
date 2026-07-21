import json
import re
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.metric import AggScore, SampleScore

DEEPSEARCH_QA_PROMPT = """
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.
""".strip()  # noqa: E501

GRADER_RATING_OUTPUT_EXAMPLE = r"""

**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""  # noqa: E501


def build_grader_prompt(question: str, reference: str, answer_type: str, response: str) -> str:
    return DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
        prompt=question,
        answer=reference,
        prompt_type=answer_type,
        response=response,
    )


def rule_fallback_score(prediction: str, reference: Any, answer_type: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    normalized_prediction = _normalize_text(prediction)
    reference_parts = _split_reference(reference)
    if answer_type == 'Single Answer':
        correct = int(any(part and _normalize_text(part) in normalized_prediction for part in reference_parts))
        expected = 1 if reference_parts else 0
        excessive = 0
    else:
        correct = sum(1 for part in reference_parts if part and _normalize_text(part) in normalized_prediction)
        expected = len(reference_parts)
        excessive = int(expected > 0 and correct < expected)
    return _score_from_counts(correct=correct, expected=expected, excessive=excessive), {
        'source': 'rule_fallback',
        'correct': correct,
        'expected': expected,
        'excessive': excessive,
    }


def parse_judge_response(response: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not response:
        return {}, {
            'empty_auto_rater_response': True,
            'error_message': 'Auto-rater response was empty.',
            'judge_raw': response,
        }

    parsed = _extract_json_object(response)
    if parsed is None:
        return _invalid_score('Invalid JSON response from auto-rater.', response)

    answer_correctness = parsed.get('Answer Correctness')
    if not isinstance(answer_correctness, dict):
        return _invalid_score("Missing or malformed 'Answer Correctness' node.", response)

    explanation = answer_correctness.get('Explanation')
    if not isinstance(explanation, str):
        return _invalid_score("Missing or malformed 'Explanation' in Answer Correctness.", response)

    details = answer_correctness.get('Correctness Details')
    if not isinstance(details, dict) or not all(isinstance(k, str) for k in details.keys()) \
            or not all(isinstance(v, bool) for v in details.values()):
        return _invalid_score("Invalid 'Correctness Details' in Answer Correctness.", response)

    excessive_answers = answer_correctness.get('Excessive Answers', [])
    if not isinstance(excessive_answers, list) or not all(isinstance(item, str) for item in excessive_answers):
        return _invalid_score("Invalid 'Excessive Answers' in Answer Correctness.", response)

    expected = len(details)
    correct = sum(1 for value in details.values() if value)
    excessive = len([answer for answer in excessive_answers if answer.strip()])

    metadata = {
        'answer_correctness_explanation': explanation,
        'correct': correct,
        'expected': expected,
        'excessive': excessive,
        'correctness_details': details,
        'excessive_answers': excessive_answers,
    }
    return _score_from_counts(correct=correct, expected=expected, excessive=excessive), metadata


def aggregate_official_scores(sample_scores: List[SampleScore]) -> List[AggScore]:
    total = len(sample_scores)
    if not total:
        return []

    valid_scores = []
    empty_model = 0
    empty_auto_rater = 0
    invalid_auto_rater = 0

    for sample_score in sample_scores:
        metadata = sample_score.score.metadata or {}
        if metadata.get('empty_model_response'):
            empty_model += 1
            continue
        if metadata.get('empty_auto_rater_response'):
            empty_auto_rater += 1
            continue
        if metadata.get('invalid_auto_rater_response'):
            invalid_auto_rater += 1
            continue
        valid_scores.append(sample_score)

    metric_values: Dict[str, List[float]] = {
        'precision': [],
        'recall': [],
        'f1_score': [],
    }
    metric_ids: Dict[str, List[Any]] = {metric_name: [] for metric_name in metric_values}

    for sample_score in valid_scores:
        for metric_name in metric_values:
            if metric_name in sample_score.score.value:
                metric_values[metric_name].append(float(sample_score.score.value[metric_name]))
                metric_ids[metric_name].append(sample_score.sample_id)

    agg_scores = []
    for metric_name, values in metric_values.items():
        if values:
            agg_scores.append(
                AggScore(
                    score=sum(values) / len(values),
                    metric_name=metric_name,
                    aggregation_name='mean',
                    num=len(values),
                    ids=metric_ids[metric_name],
                )
            )

    for metric_name, count in {
        'empty_model_response': empty_model,
        'empty_auto_rater_response': empty_auto_rater,
        'invalid_auto_rater_response': invalid_auto_rater,
    }.items():
        agg_scores.append(AggScore(
            score=count / total,
            metric_name=metric_name,
            aggregation_name='rate',
            num=total,
        ))

    return agg_scores


def _normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ''
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


def _split_reference(answer: Any) -> List[str]:
    if not answer:
        return []
    if isinstance(answer, (list, tuple)):
        return [str(part).strip() for part in answer if str(part).strip()]
    if not isinstance(answer, str):
        return [str(answer).strip()]

    answer_text = answer.strip()
    if not answer_text:
        return []
    if answer_text.startswith('[') and answer_text.endswith(']'):
        try:
            parsed = json.loads(answer_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(part).strip() for part in parsed if str(part).strip()]

    parts = [part.strip() for part in re.split(r',|;|\n', answer_text) if part.strip()]
    return parts or [answer_text]


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    json_str = text.strip()
    start_marker = '```json'
    start_idx = json_str.find(start_marker)
    if start_idx != -1:
        json_str = json_str[start_idx + len(start_marker):].strip()
        end_idx = json_str.rfind('```')
        if end_idx != -1:
            json_str = json_str[:end_idx].strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _calculate_metric(true_positives: int, false_positives: int, false_negatives: int) -> Dict[str, float]:
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }


def _score_from_counts(correct: int, expected: int, excessive: int) -> Dict[str, float]:
    return _calculate_metric(
        true_positives=correct,
        false_positives=excessive,
        false_negatives=expected - correct,
    )


def _invalid_score(error_message: str, response: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    return {}, {
        'invalid_auto_rater_response': True,
        'error_message': error_message,
        'judge_raw': response,
    }
