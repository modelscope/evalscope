# flake8: noqa: E501
"""Prompt building, answer extraction and judge helpers ported from the official PhyX evaluation.

Reference: https://github.com/NastyMarcus/PhyX (``vlmeval/dataset/utils/phyx.py``).
"""
import re
from pydantic import BaseModel
from typing import Dict, List, Optional

from evalscope.api.judge import OutputContract

# The official instruction suffixes, appended verbatim to the problem statement. The multiple-choice
# suffix is concatenated without a separator and the open-ended one with a leading space, matching
# the released ``PhyX_MC.tsv`` / ``PhyX_OE.tsv`` question strings exactly.
MC_INSTRUCTION = (
    'Please directly answer the question and provide the correct OPTION LETTER ONLY, '
    'e.g., A, B, C, D. OPTION: '
)
OE_INSTRUCTION = ' Please answer the question with step by step reasoning.'

# Every PhyX problem offers exactly these four labels.
OPTION_LABELS = ('A', 'B', 'C', 'D')

# Options ship as a single string, ``A:"...",B:"...",...``. Values are matched non-greedily up to the
# quote that precedes the next label (or the end), because option text legitimately contains commas
# and may end in a backslash. ``ast.literal_eval`` cannot be used: a value ending in a backslash
# (e.g. ``A:"1.54 \text{ m/s}^2\"``) escapes its own closing quote.
_OPTION_RE = re.compile(r'([A-Z])\s*:\s*"(.*?)"(?=\s*,\s*[A-Z]\s*:\s*"|\s*$)', re.DOTALL)

# The first ``\boxed{`` of a reply, whose content is read with brace counting rather than a regex.
_BOXED_MARKER = r'\boxed{'

# Fallback for replies that state their result in prose instead of a box.
_FINAL_ANSWER_RE = re.compile(
    r'\b(?:final\s+answer|correct\s+answer)\b[^:：]*[:：]\s*(.*?)(?=\n\n\n|\Z)',
    re.IGNORECASE | re.DOTALL,
)

# A letter the model committed to, e.g. 'The correct option is D': the first capital A-D following
# one of the answer-announcing words. Upstream enumerates the lower-case and capitalised spellings;
# the all-caps forms are added here because 'ANSWER: C' otherwise falls through to the raw reply and
# scores 0. Case-insensitivity is deliberately not expressed with a flag: ``re.IGNORECASE`` would
# also let ``[A-D]`` match the 'd' in 'derived' and invent a choice the model never made.
_ANSWER_WORDS = 'correct|answer|option|Correct|Answer|Option|CORRECT|ANSWER|OPTION'
_MC_ANSWER_RE = re.compile(rf'\b(?:{_ANSWER_WORDS})\b[\s\S]*?([A-D])')

# Replies that echo the option list instead of announcing a choice, e.g. 'B: 5.2 mW/cm^2'.
_MC_LABEL_RE = re.compile(r'([ABCD]):')


class Judgment(BaseModel):
    reasoning: str = ''
    verdict: bool


VERDICT_CONTRACT = OutputContract(schema_model=Judgment)

# LaTeX spellings normalised before string comparison, as the official evaluator does. Applied to the
# prediction only -- the ground truth is left untouched, matching upstream.
_LATEX_SUBSTITUTIONS = {r'\dfrac': r'\frac', r'\pi': '3.14'}

_ICE_EXAMPLES_OE = [
    """
Ground truth answer: 502 \n
Predicted answer: The mass of block (B) is:
[
\\boxed{ 50 \\sqrt{101} }
] \n
{"verdict": true}
""",
    """
Ground truth answer: 46.3 kN \n
Predicted answer: The tension ( T_B ) in the cable is approximately:
[
\\boxed{46300 }
] \n
{"verdict": true}
""",
    """
Ground truth answer: 12 m/s \n
Predicted answer: The speed of the box after 2.00 seconds is:
[
\\boxed{11.3, \\text{m/s}}
] \n
{"verdict": false}
""",
    """
Ground truth answer: 36.00 kg \n
Predicted answer: The mass of the hanging block ( m_2 ) must be approximately:
[
\\boxed{36.1, \\text\\{kg\\}}
] \n
{"verdict": true}
""",
    """
Ground truth answer: 3.2 m \n
Predicted answer: The stuntman and villain slide approximately \\frac\\{10\\}{3.1415} meters**.
{"verdict": true}
""",
]

_ICE_EXAMPLES_MC = [
    """
Ground truth answer: A \n
Predicted answer: A \n
{"verdict": true}
""",
    """
Ground truth answer: B \n
Predicted answer: A \n
{"verdict": false}
""",
    """
Ground truth answer: C \n
Predicted answer: ### Step 1: Calculate ( l_1 )
The lightbulb is ( 2.50, \\text\\{m\\}) above the floor, and the bottom of the mirror is (0.50, \\text\\{m\\}) above the floor. The vertical distance from the lightbulb to the bottom of the mirror is:
[
\\Delta y_1 = 2.50, \\text\\{m\\} - 0.50, \\text\\{m\\} = 2.00, \\text\\{m\\}.
] \n
{"verdict": false}
""",
    """
Ground truth answer: D \n
Predicted answer: The correct option is D. \n
{"verdict": true}
""",
]

_OE_JUDGE_TASK = """
Please read the following example. Given predicted answer and ground truth answer,
compare the these two answers, then decide whether they are matched or unmatched.
If the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If the given predicted mentions "approximately", then allow the Approximation Error, such as 0.49 and approximately 0.5, 0.81 and approximately 0.8. \n
"""

_MC_JUDGE_TASK = """
Please read the following example. Given predicted answer and ground truth answer for Multi-Choice question.
The ground truth answer would be A/B/C/D. The predicted answer would be some words containing A/B/C/D.
Please compare the these two answers, then decide whether they are matched or unmatched. \n
"""


def parse_options(raw_options: str) -> Dict[str, str]:
    """Parse the raw ``A:"...",B:"..."`` option string into a label -> text mapping."""
    return dict(_OPTION_RE.findall(raw_options.strip()))


def build_mc_question(description: str, question: str, options: Dict[str, str]) -> str:
    """Build the official multiple-choice prompt: description, question, instruction, options."""
    rendered_options = ' '.join(f'{label}: {text}' for label, text in options.items())
    return f'{description} {question}{MC_INSTRUCTION}{rendered_options}'


def build_oe_question(description: str, question: str) -> str:
    """Build the official open-ended prompt: description, question, reasoning instruction."""
    return f'{description} {question}{OE_INSTRUCTION}'


def _normalize_latex(text: str) -> str:
    """Apply the official LaTeX spelling substitutions used before string comparison."""
    for old, new in _LATEX_SUBSTITUTIONS.items():
        text = text.replace(old, new)
    return text


def extract_boxed_content(text: str) -> Optional[str]:
    """Return the content of the first ``\\boxed{...}``, or None when there is none.

    An unterminated ``\\boxed{`` yields None rather than the remainder of the reply: a truncated
    box carries no complete answer, and returning its prefix would score a partial expression.
    """
    start = text.find(_BOXED_MARKER)
    if start == -1:
        return None
    rest = text[start + len(_BOXED_MARKER):]
    depth = 0
    for index, char in enumerate(rest):
        if char == '{':
            depth += 1
        elif char == '}':
            if depth == 0:
                return rest[:index]
            depth -= 1
    return None


def extract_oe_answer(prediction: str) -> str:
    """Extract the final answer of an open-ended reply.

    Prefers a boxed answer, then a 'final answer:' / 'correct answer:' statement. A reply doing
    neither is returned unchanged, which is what the official evaluator compares against.
    """
    prediction = prediction.strip()
    boxed = extract_boxed_content(prediction)
    if boxed is not None:
        return _normalize_latex(boxed).strip()

    match = _FINAL_ANSWER_RE.search(prediction)
    if match:
        return _normalize_latex(match.group(1)).strip()
    return prediction


def extract_mc_answer(prediction: str) -> str:
    """Extract the chosen option label of a multiple-choice reply.

    A reply that announces no label is returned unchanged, so that the substring fallbacks in
    ``match_mc_answer`` still see the full text.
    """
    prediction = prediction.strip()
    match = _MC_ANSWER_RE.search(prediction)
    if match:
        return match.group(1)

    labels = _MC_LABEL_RE.findall(prediction)
    if labels:
        return labels[-1]
    return prediction


def match_oe_answer(extracted: str, prediction: str, reference: str) -> bool:
    """Official string-level match for open-ended answers.

    The reference is stripped before the containment test, which upstream does not do. 175 of the
    3,000 ground-truth values carry a trailing space, and requiring the model to reproduce it would
    reject a reply that ends in exactly the right value. The containment test stays case-sensitive:
    physical units are case-bearing, so folding case would equate ``7.55N`` with ``7.55n``.
    """
    reference = reference.strip()
    return (
        reference.lower() == extracted.strip().lower() or reference.lower() == prediction.strip().lower()
        or reference in prediction
    )


def match_mc_answer(extracted: str, prediction: str, reference: str) -> bool:
    """Official string-level match for multiple-choice answers.

    Besides the extracted label, the reply is accepted when it marks the correct label the way the
    options are printed (``D:``) or emphasised (``**D``), covering replies that restate the option
    instead of announcing a letter.
    """
    reference = reference.strip()
    if reference.lower() == extracted.strip().lower():
        return True
    return f'{reference}:' in prediction or f'{reference}**' in prediction or f'**{reference}' in prediction


def build_oe_judge_prompt(prediction: str, reference: str) -> str:
    """Build the official open-ended judge prompt (in-context examples + the pair to compare)."""
    return _build_judge_prompt(_OE_JUDGE_TASK, _ICE_EXAMPLES_OE, prediction, reference)


def build_mc_judge_prompt(prediction: str, reference: str) -> str:
    """Build the official multiple-choice judge prompt."""
    return _build_judge_prompt(_MC_JUDGE_TASK, _ICE_EXAMPLES_MC, prediction, reference)


def _build_judge_prompt(task_description: str, examples: List[str], prediction: str, reference: str) -> str:
    prompt = task_description
    for example in examples:
        prompt += example + '\n'
    prompt += f'Ground truth answer: {reference} \n'
    prompt += f'Predicted answer: {prediction} \n'
    return prompt
