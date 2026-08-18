# flake8: noqa: E501
"""Helpers for the HiPhO physics Olympiad benchmark: prompts, marking-scheme
parsing, boxed-answer extraction, and judge-response parsing."""
import re
from typing import List

# Exams whose problems are written in Chinese; every other exam is English.
# Matches the official language mapping in the HiPhO paper (Appendix B.1).
CHINESE_EXAMS = ('CPhO', 'PanMechanics')

# Official inference prompt for English-language exams (HiPhO paper, Appendix B.1).
ENGLISH_PROMPT = """You are participating in a high school physics Olympiad exam.
Please read the following question carefully and provide a clear, step-by-step solution with full reasoning.
Instructions:
1. Use LaTeX to format all variables, equations, and calculations.
2. Enclose your full reasoning process within <think></think> tags.
3. Provide the final answer within <answer></answer> tags, using the format of [\\boxed{{answer}}]. Do not include units inside the box.
4. For multiple sub-questions, list the answers in order using the format: [\\boxed{{answer1}}, \\boxed{{answer2}}, …].
5. For multiple-choice questions, provide the final selected option(s) in the boxed answer instead of the calculation result (e.g., [\\boxed{{A}}]).
Example of Output:
<think>
Step 1: Analyze the problem… Step 2: Apply the relevant equations…
</think>
<answer>
[\\boxed{{A}}, \\boxed{{3.2}}]
</answer>
Useful information (formulas, constants, units, if applicable):
{information}
Context (if applicable):
{context}
Question (Answer only the question stated below):
{question}"""

# Official inference prompt for Chinese-language exams (HiPhO paper, Appendix B.1).
CHINESE_PROMPT = """你正在参加高中物理竞赛。
请仔细阅读下列题目，结合上下文信息，详细推导并给出清晰、有条理的解题步骤与完整的逻辑推理过程。
作答要求：
1. 所有物理量、公式和计算过程须使用 LaTeX 格式书写。
2. 将完整的推理过程用 <think>和 </think>标签括起来。
3. 将最终答案置于 <answer>和 </answer>标签中，答案格式为 [\\boxed{{答案}}]，方框内不包含单位。
4. 对于包含多个小问的题目，按顺序列出所有答案，格式为：[\\boxed{{答案1}}, \\boxed{{答案2}}, …]。
5. 对于选择题，请在答案的方框中给出最终选择的选项，而不是计算结果（例如：[\\boxed{{A}}]）。
输出示例：
<think>
第一步：分析问题… 第二步：运用相关公式…
</think>
<answer>
[\\boxed{{A}}, \\boxed{{3.2}}]
</answer>
可用信息（如物理公式、常数、单位等）：
{information}
背景信息（如有）：
{context}
题目内容（仅回答以下问题）：
{question}"""

# Answer-level judge, adapted from the PHYSICS design (HiPhO paper, Appendix B.2).
ANSWER_JUDGE_PROMPT = """You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
Special considerations:
1. Multiple Answers: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].
2. Mathematical Problems: If the formats differ but the answers are mathematically equivalent such as 256/55=4.65, respond with [Correct].
3. Physics Problems: If the values match such as 3=3 GHz, respond with [Correct].
4. Explicit Options: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.
5. No Explicit Options: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
Question: {question}
Output sentence: {given_answer}
Correct answer: {ground_truth}
Judge whether the output sentence correctly answers the question.
""" ''

# Step-level judge (HiPhO paper, Appendix B.3). The criterion text itself states the
# points to award, so the judge returns the awarded points as a single number.
STEP_JUDGE_PROMPT = """You are an expert physics competition grader. Evaluate the student's solution against the specific grading criterion.
Physics Problem:
{question}
Student's Solution:
{prediction}
Grading Criterion:
{criterion}
Instructions:
1. Analyze the student's solution for physics concepts, mathematical derivations, and calculations.
2. Award points strictly according to the criterion.
3. Consider both conceptual understanding and technical accuracy.
Award points strictly according to the criterion.
""" ''

# Every criterion states its own allocation, e.g. "Award 0.1 pt if ..." or "得 0.5 分".
_CRITERION_POINTS_RE = re.compile(
    r'(?:award|給|给|得|扣)\s*\$?\s*([0-9]*\.?[0-9]+)\s*\$?\s*(?:pts?|points?|分)',
    re.IGNORECASE,
)

# Sentinel that ``LLMJudge.judge`` returns instead of raising on a failed request.
JUDGE_ERROR_PREFIX = '[ERROR]'


def is_chinese_exam(source: str) -> bool:
    """Return True when the exam's problems are written in Chinese."""
    return source.startswith(CHINESE_EXAMS)


def normalize_marking(marking) -> List[List[str]]:
    """Normalize a record's ``marking`` field into a list of grading schemes.

    A scheme is a list of criterion strings. Some records use a single flat list
    of criteria, others nest one list per alternative official scheme (EuPhO /
    NBPhO), and some ship no marking at all (answer-level exams).
    """
    if not marking:
        return []
    if all(isinstance(item, str) for item in marking):
        return [list(marking)]
    return [[item] if isinstance(item, str) else list(item) for item in marking]


def criterion_points(criterion: str) -> float:
    """Extract the maximum points a criterion can award from its description.

    A criterion may mention several tiers (e.g. full vs. partial credit), so the
    largest stated value is the criterion's maximum.
    """
    values = [float(v) for v in _CRITERION_POINTS_RE.findall(criterion)]
    return max(values) if values else 0.0


def extract_boxed_answers(text: str) -> List[str]:
    """Extract the contents of every complete ``\\boxed{...}`` in order of appearance.

    Brace matching is used so nested braces inside a boxed expression (e.g.
    ``\\boxed{\\frac{1}{2}}``) are captured correctly. A trailing unbalanced
    ``\\boxed{`` is ignored rather than reported as a partial answer: it means the
    reply was truncated mid-answer, and emitting the fragment would both invent an
    answer and shift the ordered alignment that answer-level scoring relies on.
    """
    answers: List[str] = []
    idx = 0
    needle = r'\boxed'
    while True:
        pos = text.find(needle, idx)
        if pos == -1:
            break
        brace = text.find('{', pos)
        if brace == -1:
            break
        depth = 0
        content = []
        i = brace
        while i < len(text):
            char = text[i]
            if char == '{':
                depth += 1
                if depth == 1:
                    i += 1
                    continue
            elif char == '}':
                depth -= 1
                if depth == 0:
                    break
            content.append(char)
            i += 1
        if depth != 0:
            # Unterminated brace: the rest of the reply is truncated, so stop here.
            break
        answers.append(''.join(content).strip())
        idx = i + 1
    return answers


def strip_boxed(text: str) -> str:
    """Return the inner expression of a single ``\\boxed{...}`` payload if present."""
    boxed = extract_boxed_answers(text)
    if boxed:
        return boxed[-1]
    return text.strip()
