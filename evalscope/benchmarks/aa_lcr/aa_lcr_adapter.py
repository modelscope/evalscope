# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import re
from pathlib import Path
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags, HubType
from evalscope.utils.logger import get_logger
from evalscope.api.metric import Score
logger = get_logger()

# Default judge prompt template
JUDGE_PROMPT = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT. For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {correct_answer}
CANDIDATE ANSWER TO ASSESS: {response}

Reply only with CORRECT or INCORRECT."""

PROMPT_TEMPLATE = """
BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr',
        pretty_name='AA-LCR',
        tags=[Tags.KNOWLEDGE, Tags.REASONING],
        description=
        'AA-LCR (Artificial Analysis Long Context Retrieval) is a benchmark for evaluating long-context '
        'retrieval and reasoning capabilities of language models across multiple documents.',  # noqa: E501
        dataset_id='ArtificialAnalysis/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}',
        extra_params={
            'text_dir': None
        }
    )
)
class AALCRAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._use_llm_judge = True
        
        # Get extra parameters
        self.text_dir = self.extra_params.get('text_dir')
        if not self.text_dir or not Path(self.text_dir).exists():
            raise ValueError(
                "Please download and extract the AA-LCR documents from "
                "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/tree/main/extracted_text, "
                "and set 'text_dir' in extra_params accordingly."
            )
        self.text_dir = Path(self.text_dir)
        self.dataset_hub = HubType.HUGGINGFACE

    def _get_context(self, record: Dict[str, Any]) -> str:
        doc_folder = self.text_dir / record['document_category'] / record['document_set_id']
        
        # Check if the document folder exists
        if not doc_folder.exists() or not doc_folder.is_dir():
            logger.warning(f"Document folder not found: {doc_folder}. Returning empty context.")
            return ""

        doc_blocks = []
        try:
            for file_path in doc_folder.iterdir():
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8').strip()
                        if content:
                            doc_blocks.append(content)
                    except (IOError, UnicodeDecodeError) as e:
                        logger.warning(f"Could not read file {file_path}, skipping: {e}")
        except OSError as e:
            logger.warning(f"Could not access document folder {doc_folder}: {e}")
            return f"ERROR: Could not read documents for {record['document_category']}/{record['document_set_id']}"
            
        documents_text = "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(doc_blocks))
        return documents_text

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a record to a Sample with long-context prompt."""
        context = self._get_context(record)
        prompt = PROMPT_TEMPLATE.format(
            documents_text=context,
            question=record['question'])
        
        return Sample(
            input=prompt,
            target=record['answer'],
            metadata={
                'question': record['question'],
                'data_source_urls': record['data_source_urls'],
                'input_tokens': record.get('input_tokens', 0),
            }
        )

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

        judge_prompt = JUDGE_PROMPT.format(
            question=task_state.metadata['question'], correct_answer=reference, response=filtered_prediction
        )

        # Request judge and obtain score
        judge_response = self.llm_judge.judge(prompt=judge_prompt)

        # Parse judge response to get accuracy score
        # Use word boundaries to avoid matching "CORRECT" within "INCORRECT"
        is_correct = bool(re.search(r'\bCORRECT\b', judge_response, re.IGNORECASE))
        score.value = {
            'acc': 1.0 if is_correct else 0.0,
        }
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
        }
        score.main_score_name = 'acc'
        return score
