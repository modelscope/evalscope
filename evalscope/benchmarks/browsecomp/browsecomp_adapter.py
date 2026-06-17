import base64
import csv
import hashlib
import io
import os
import re
import requests
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.dataset.loader import DictDataLoader
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import csv_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()

BROWSECOMP_DATA_URL = 'https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv'

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

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

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


def parse_judge_response(response: Any) -> bool:
    """Return True only when the judge explicitly reports correctness."""
    if not response or not isinstance(response, str):
        return False
    match = re.search(r'correct:\s*["\']?(yes|no)["\']?', response, re.IGNORECASE)
    return bool(match and match.group(1).lower() == 'yes')


@register_benchmark(
    BenchmarkMeta(
        name='browsecomp',
        pretty_name='BrowseComp',
        tags=[Tags.AGENT, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

BrowseComp is an OpenAI benchmark for evaluating browsing and search agents. It contains 1,266 hard-to-find, fact-seeking questions with short, verifiable answers.

## Task Description

- **Task Type**: Search-agent factual question answering
- **Input**: Challenging natural-language question that generally requires persistent web browsing
- **Output**: Explanation, exact answer, and confidence
- **Grading**: LLM judge compares the final answer against the reference answer

## Key Features

- Tests persistence, creative search, and multi-hop evidence gathering
- Uses short answers to keep grading tractable
- Official data is distributed as encrypted CSV rows and decrypted at evaluation time
- Compatible with EvalScope agent configuration via the default agent loop path

## Evaluation Notes

- Default evaluation uses the official public BrowseComp CSV.
- Use `dataset_args={'browsecomp': {'local_path': '/path/to/browse_comp_test_set.csv'}}` to evaluate from a local copy.
- The primary metric is `is_correct`; `is_incorrect` is also reported.
- LLM judge is enabled by default. `JudgeStrategy.RULE` falls back to normalized exact match.
""",  # noqa: E501
        dataset_id=BROWSECOMP_DATA_URL,
        metric_list=['is_correct', 'is_incorrect'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=QUERY_TEMPLATE,
        paper_url='https://arxiv.org/abs/2504.12516',
    )
)
class BrowseCompAdapter(DefaultDataAdapter):
    """Adapter for the BrowseComp browsing-agent benchmark."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._use_llm_judge = True
        self._suppress_doc_sample_example = True

    def load(self) -> tuple[DatasetDict, None]:
        rows = self._load_rows(self.dataset_id)
        dataset = DictDataLoader(
            dict_list=rows,
            sample_fields=self.record_to_sample,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        ).load()
        return DatasetDict({'default': dataset}), None

    def _load_rows(self, data_id_or_path: str) -> List[Dict[str, Any]]:
        if os.path.isdir(data_id_or_path):
            data_id_or_path = self._find_csv_file(data_id_or_path)

        if os.path.isfile(data_id_or_path):
            logger.info(f'Loading BrowseComp dataset from local file: {data_id_or_path}')
            return csv_to_list(data_id_or_path)

        if not data_id_or_path.startswith(('http://', 'https://')):
            raise FileNotFoundError(f'Local file or directory not found: {data_id_or_path}')

        logger.info(f'Loading BrowseComp dataset from URL: {data_id_or_path}')
        response = requests.get(data_id_or_path, timeout=60)
        response.raise_for_status()
        return list(csv.DictReader(io.StringIO(response.text)))

    @staticmethod
    def _find_csv_file(data_dir: str) -> str:
        candidates = [
            'browse_comp_test_set.csv',
            'default_test.csv',
            'default.csv',
            'test.csv',
        ]
        for candidate in candidates:
            path = os.path.join(data_dir, candidate)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f'No BrowseComp CSV file found in {data_dir}.')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        canary = record.get('canary') or ''
        question = decrypt(record.get('problem') or '', canary)
        answer = decrypt(record.get('answer') or '', canary)

        return Sample(
            input=question,
            target=answer,
            metadata={
                'problem_topic': record.get('problem_topic') or '',
                'canary': canary,
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

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        metadata = task_state.metadata or {}
        question = metadata.get('question') or task_state.input_text
        prompt = GRADER_TEMPLATE.format(
            question=question,
            response=original_prediction,
            correct_answer=reference,
        )
        judge_response = self.llm_judge.judge(prompt)
        is_correct = parse_judge_response(judge_response)

        score.value = {
            'is_correct': 1.0 if is_correct else 0.0,
            'is_incorrect': 0.0 if is_correct else 1.0,
        }
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
        }
        score.main_score_name = 'is_correct'
        return score
