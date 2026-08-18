import re
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Tuple

from evalscope.api.judge import OutputContract
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


def rule_fallback_score(prediction: str, reference: str, answer_type: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    normalized_prediction = _normalize_text(prediction)
    if answer_type == 'Single Answer':
        reference_parts = [reference.strip()] if reference.strip() else []
        correct = int(any(part and _normalize_text(part) in normalized_prediction for part in reference_parts))
        expected = 1 if reference_parts else 0
        excessive = 0
    else:
        reference_parts = _split_reference(reference)
        correct = sum(1 for part in reference_parts if part and _normalize_text(part) in normalized_prediction)
        expected = len(reference_parts)
        excessive = 0
    return _score_from_counts(correct=correct, expected=expected, excessive=excessive), {
        'source': 'rule_fallback',
        'correct': correct,
        'expected': expected,
        'excessive': excessive,
    }


class AnswerCorrectness(BaseModel):
    """The judge's ``Answer Correctness`` node, using the official DeepSearchQA keys."""

    model_config = ConfigDict(populate_by_name=True)

    explanation: str = Field(alias='Explanation')
    correctness_details: Dict[str, bool] = Field(alias='Correctness Details')
    excessive_answers: List[str] = Field(default_factory=list, alias='Excessive Answers')


class Grade(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    answer_correctness: AnswerCorrectness = Field(alias='Answer Correctness')


GRADE_CONTRACT = OutputContract(schema_model=Grade)


def metrics_from_grade(grade: Grade) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Fold the judge's correctness verdict into precision/recall/f1 plus diagnostic metadata."""
    details = grade.answer_correctness.correctness_details
    excessive_answers = grade.answer_correctness.excessive_answers
    expected = len(details)
    correct = sum(1 for value in details.values() if value)
    excessive = len([answer for answer in excessive_answers if answer.strip()])

    metadata = {
        'answer_correctness_explanation': grade.answer_correctness.explanation,
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
    judge_failure = 0

    for sample_score in sample_scores:
        metadata = sample_score.score.metadata or {}
        if metadata.get('empty_model_response'):
            empty_model += 1
            continue
        # A judge that could not produce a usable verdict (empty or malformed) is excluded from the
        # means and reported as a single failure rate, rather than the old empty/invalid split.
        if not sample_score.score.status.is_usable:
            judge_failure += 1
            continue
        valid_scores.append(sample_score)

    metric_values: Dict[str, List[float]] = {
        'precision': [],
        'recall': [],
        'f1': [],
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
                    aggregation='mean',
                    num=len(values),
                    ids=metric_ids[metric_name],
                )
            )

    for metric_name, count in {
        'empty_model_response': empty_model,
        'judge_parse_failure': judge_failure,
    }.items():
        agg_scores.append(AggScore(
            score=count / total,
            metric_name=metric_name,
            aggregation='rate',
            num=total,
        ))

    return agg_scores


def _normalize_text(text: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


def _split_reference(answer: str) -> List[str]:
    if not answer:
        return []
    answer_text = answer.strip()
    if not answer_text:
        return []
    return [part.strip() for part in re.split(r',|;|\n', answer_text) if part.strip()]


def _calculate_metric(true_positives: int, false_positives: int, false_negatives: int) -> Dict[str, float]:
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
    }


def _score_from_counts(correct: int, expected: int, excessive: int) -> Dict[str, float]:
    return _calculate_metric(
        true_positives=correct,
        false_positives=excessive,
        false_negatives=expected - correct,
    )
