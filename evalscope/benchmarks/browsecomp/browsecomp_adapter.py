import base64
import hashlib
import re
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags

BROWSECOMP_DATASET_ID = 'evalscope/browse_comp'


# The grader template requires a "correct: yes" / "correct: no" line.
class Grade(BaseModel):
    extracted_final_answer: str = ''
    reasoning: str = ''
    correct: Literal['yes', 'no']
    confidence: int = Field(default=100, ge=0, le=100)


GRADE_CONTRACT = OutputContract(schema_model=Grade)

QUERY_TEMPLATE = """
{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()  # noqa: E501

ANSWER_PATTERN = re.compile(r'(?im)^\s*Exact Answer:\s*(.+?)\s*$')


def derive_key(password: str, length: int) -> bytes:
    """Derive the XOR key used by the official BrowseComp release."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[:length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt a base64-encoded BrowseComp field with the row canary."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def normalize_answer(answer: Any) -> str:
    """Normalize short factual answers for rule-based exact matching."""
    if not answer or not isinstance(answer, str):
        return ''
    return ' '.join(re.sub(r'[^\w\s]', ' ', answer.lower()).split())


@register_benchmark(
    BenchmarkMeta(
        name='browsecomp',
        pretty_name='BrowseComp',
        tags=[Tags.AGENT, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

BrowseComp is an OpenAI benchmark for evaluating browsing and search agents. It contains 1,266 hard-to-find, fact-seeking questions with short, verifiable answers. EvalScope loads the mirrored dataset from ModelScope (`evalscope/browse_comp`).

## Task Description

- **Task Type**: Search-agent factual question answering
- **Input**: Challenging natural-language question that generally requires persistent web browsing
- **Output**: Explanation, exact answer, and confidence
- **Grading**: LLM judge compares the final answer against the reference answer

## Key Features

- Tests persistence, creative search, and multi-hop evidence gathering
- Uses short answers to keep grading tractable
- Official data is distributed as encrypted CSV rows and decrypted at evaluation time
- Classified as an Agent benchmark and compatible with EvalScope agent loop modes
- Supports single-turn model evaluation by default and native/external agent execution when `TaskConfig.agent_config` is provided

## Evaluation Notes

- Default evaluation loads `evalscope/browse_comp` from ModelScope through the standard EvalScope dataset loader.
- Use `TaskConfig.agent_config` to evaluate BrowseComp with EvalScope agent loop capabilities such as native tool-use or external agent runners.
- The primary metric is `is_correct`; `is_incorrect` is also reported.
- LLM judge is enabled by default. `JudgeStrategy.RULE` falls back to normalized exact match.
""",  # noqa: E501
        dataset_id=BROWSECOMP_DATASET_ID,
        metric_list=['is_correct', 'is_incorrect'],
        primary_metric='is_correct',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=QUERY_TEMPLATE,
        paper_url='https://arxiv.org/abs/2504.12516',
    )
)
class BrowseCompAdapter(AgentAdapter):
    """Adapter for the BrowseComp browsing-agent benchmark."""
    scoring_policy = ScoringPolicy.JUDGE_DEFAULT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._suppress_doc_sample_example = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        canary = record.get('canary') or ''
        question = decrypt(record.get('problem') or '', canary)
        answer = decrypt(record.get('answer') or '', canary)

        return Sample(
            input=question,
            target=answer,
            metadata={
                'problem_topic': record.get('problem_topic') or '',
                'question': question,
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        match = ANSWER_PATTERN.search(prediction)
        if match:
            return match.group(1).strip()
        return prediction.strip()

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        is_correct = normalize_answer(filtered_prediction) == normalize_answer(reference)
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                'is_correct': 1.0 if is_correct else 0.0,
                'is_incorrect': 0.0 if is_correct else 1.0,
            },
            main_score_name='is_correct',
        )
        score.metadata = {'source': 'rule_exact_match'}
        return score

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            task_state = judge_context.task_state
            prompt = GRADER_TEMPLATE.format(
                question=(task_state.metadata or {}).get('question') or task_state.input_text,
                response=judge_context.original_prediction,
                correct_answer=judge_context.reference,
            ) + case.output_contract.instruction()
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            is_correct = case_verdicts[0].value.correct == 'yes'
            return ReducedVerdict(
                value={
                    'is_correct': 1.0 if is_correct else 0.0,
                    'is_incorrect': 0.0 if is_correct else 1.0
                }
            )

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='grade', output_contract=GRADE_CONTRACT)],
            request=request,
            reduce=reduce,
            main_score_name='is_correct'
        )
