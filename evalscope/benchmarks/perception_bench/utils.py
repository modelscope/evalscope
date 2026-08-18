# flake8: noqa: E501
"""Utilities for the PerceptionBench judge protocol.

Ports the official grading prompt and verdict parsing from
https://github.com/MoonshotAI/PerceptionBench (``eval/judge_prompt.txt`` and ``eval/eval.py``).
"""
import re

JUDGE_TEMPLATE = """Please act as a professional teacher and grade the student's answer. Below are the question, the student's answer, and the reference answer. Based on the question and the reference answer, analyze the student's answer and judge whether it correctly answers the question. I will provide several examples to help you understand how to judge.

========== [Notice] ==========
0. You only need to compare the consistency between the reference answer and the student's answer, focusing especially on the content after summarizing phrases such as "Final answer:". The reference answer is absolutely correct; judge solely by whether the student's final result matches it, even if the solution process is correct.
1. If the question contains multiple sub-questions, output the student's answer, the reference answer, and the consistency for each sub-question in "reasoning"; judge correct only when all sub-questions are consistent.
2. For multiple-answer questions, the student's answer must contain all correct answers without any extra ones. Pay special attention when the reference answer contains "or": decide whether all answers must be given based on the specific question.
3. If you believe the student's answer equals the reference answer after simplification, you must provide the complete simplification process in "reasoning" until the two are equal; otherwise it cannot be judged correct. You may not vaguely claim that "they can be made equal".
4. For numerical answers, integers or values with at most 4 significant figures must match exactly; values with more than 4 significant figures must match within 4 significant figures. In "reasoning", convert both to standard scientific notation, first compare the order of magnitude, then compare the first 4 significant figures. If the units differ, convert to a common unit first.
5. For English writing questions, if the student does not answer in English, judge it incorrect.
6. For physics, chemistry, and biology questions, if the reference answer contains a technical term, the student's answer must contain that exact term; synonyms are not accepted.
7. When the question asks to explain a term, redundant explanation is not penalized, but missing key points must be judged incorrect.
8. For multiple-choice questions, judge incorrect whenever the student's selected option differs from the reference answer, regardless of the content.

========== [Examples] ==========
========== [Example 1.1] ==========
Question: omitted    Student answer: 34    Reference answer: (1) 12; (2) 34
{"reasoning": "The student answer 34 only addresses the second sub-question and leaves the first unanswered.", "verdict": false}
========== [Example 1.2] ==========
Question: omitted    Student answer: 13; 34    Reference answer: (1) 12; (2) 34
{"reasoning": "Sub-question 1: student 13 vs. reference 12, inconsistent. Sub-question 2: student 34 vs. reference 34, consistent. Overall, incorrect.", "verdict": false}
========== [Example 2.1] ==========
Question: omitted    Student answer: A    Reference answer: ABC
{"reasoning": "The student answer A is incomplete; the correct answer is ABC.", "verdict": false}
========== [Example 2.2] ==========
Question: omitted    Student answer: ABC    Reference answer: AC
{"reasoning": "The student answer ABC includes an extra B; the correct answer is AC.", "verdict": false}
========== [Example 3.1] ==========
Question: omitted    Student answer: y''(0) = 13e^2 - e    Reference answer: y''(0) = 12e^2 - 1
{"reasoning": "The student answer differs in form from the reference and cannot be made equal through simplification.", "verdict": false}
========== [Example 3.2] ==========
Question: omitted    Student answer: y''(0) = 12e^2 - e    Reference answer: y''(0) = (12e - 1)e
{"reasoning": "12e^2 - e = (12e - 1)e, which is equivalent to the reference answer.", "verdict": true}
========== [Example 4.1] ==========
Question: omitted    Student answer: 349    Reference answer: 342
{"reasoning": "Integer answers must match exactly; 349 differs from 342.", "verdict": false}
========== [Example 4.2] ==========
Question: omitted    Student answer: 0.325    Reference answer: 0.618
{"reasoning": "In scientific notation, 0.325 = 3.250e-1 and 0.618 = 6.180e-1; the exponents match but the first significant figure differs.", "verdict": false}
========== [Example 4.3] ==========
Question: omitted    Student answer: 48.67    Reference answer: 48.675
{"reasoning": "48.67 = 4.867e1 and 48.675 = 4.868e1 (to 4 significant figures); the exponents match but the 4th significant figure differs.", "verdict": false}
========== [Example 4.4] ==========
Question: omitted    Student answer: 4.85e4    Reference answer: 485
{"reasoning": "4.85e4 vs. 485 = 4.85e2; the two differ in order of magnitude.", "verdict": false}
========== [Example 8.1] ==========
Question: A. 11  B. 22  C. 33  D. 44    Reference answer: A. 11    Student answer: C. 11
{"reasoning": "The student option C differs from the reference option A.", "verdict": false}
========== [Example 8.2] ==========
Question: A. 11  B. 22  C. 33  D. 44    Reference answer: B    Student answer: C
{"reasoning": "The student option C differs from the reference option B.", "verdict": false}
========== [Notice] ==========
0. You only need to compare the consistency between the reference answer and the student's answer, focusing especially on the content after summarizing phrases such as "Final answer:". The reference answer is absolutely correct; judge solely by whether the student's final result matches it, even if the solution process is correct.
1. If the question contains multiple sub-questions, output the student's answer, the reference answer, and the consistency for each sub-question in "reasoning"; judge correct only when all sub-questions are consistent.
2. For multiple-answer questions, the student's answer must contain all correct answers without any extra ones. Pay special attention when the reference answer contains "or": decide whether all answers must be given based on the specific question.
3. If you believe the student's answer equals the reference answer after simplification, you must provide the complete simplification process in "reasoning" until the two are equal; otherwise it cannot be judged correct. You may not vaguely claim that "they can be made equal".
4. For numerical answers, integers or values with at most 4 significant figures must match exactly; values with more than 4 significant figures must match within 4 significant figures. In "reasoning", convert both to standard scientific notation, first compare the order of magnitude, then compare the first 4 significant figures. If the units differ, convert to a common unit first.
5. For English writing questions, if the student does not answer in English, judge it incorrect.
6. For physics, chemistry, and biology questions, if the reference answer contains a technical term, the student's answer must contain that exact term; synonyms are not accepted.
7. When the question asks to explain a term, redundant explanation is not penalized, but missing key points must be judged incorrect.
8. For multiple-choice questions, judge incorrect whenever the student's selected option differs from the reference answer, regardless of the content.

Now you may begin grading.
========== [Question] ==========
{problem}
========== [Student Answer] ==========
{assistant_answer}
========== [Reference Answer] ==========
{reference_answer}
========== [Your Judgment] ==========
"""


def normalize_escape(text: str) -> str:
    r"""Normalize backslash escapes before feeding text to the judge.

    Kept behaviourally identical to ``normalize_escape`` in the official evaluator so
    that the judge sees exactly the official prompt: a run of backslashes followed by a
    space, digit or hyphen collapses to two backslashes plus a space (the trailing
    whitelisted character is intentionally dropped), any other backslash run collapses
    to a single backslash, and a literal ``\n`` sequence becomes a real newline.  The
    same transform is applied to question, reference and prediction alike, so the lossy
    whitelist branch cannot bias grading.
    """
    whitelist = ' 0123456789-'
    text = re.sub(rf'\\+[{whitelist}]', r'\\\\' + ' ', text)
    text = re.sub(rf'\\+(?![{whitelist}])', r'\\', text)
    return re.sub(r'\\+n', '\n', text)


def build_judge_prompt(question: str, prediction: str, reference: str) -> str:
    """Build the official teacher-grading prompt for a single sample."""
    prompt = JUDGE_TEMPLATE.replace('{problem}', normalize_escape(question).strip())
    prompt = prompt.replace('{reference_answer}', normalize_escape(reference).strip())
    return prompt.replace('{assistant_answer}', normalize_escape(prediction).strip())
