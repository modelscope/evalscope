import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from evalscope.agent.environments.local import LocalAgentEnvironment

BINARY_SYSTEM_PROMPT = """You are an expert evaluator tasked with assessing whether a document satisfies specific rubric criteria. Your evaluation must be precise, objective, and based solely on the evidence present in the document.

## Evaluation Framework

You will evaluate each rubric criterion using a binary satisfaction scale:

1. **Not Satisfied (Score: 0.0)**: The document fails to meet the criterion. Key elements are missing, incorrect, or inadequately addressed.

2. **Satisfied (Score: 1.0)**: The document fully meets the criterion. All required elements are present, well-developed, and appropriately detailed.

## Evaluation Process

1. **Understand the Criterion**: Carefully read and interpret what the rubric is asking for.
2. **Search for Evidence**: Systematically review the document for relevant content that addresses the criterion.
3. **Assess Completeness**: Evaluate whether the evidence satisfies or fails to satisfy the criterion.
4. **Provide Reasoning**: Explain your evaluation with specific references to the document content.

## Important Guidelines

- Base your evaluation ONLY on what is explicitly present in the document
- Do not make assumptions about implied or missing content
- Consider the quality, completeness, and relevance of the evidence
- Be consistent in your evaluation standards across all criteria
- Provide specific examples from the document to support your verdict

Note: Example lists in these rubrics are intended to illustrate possible reasoning patterns or relevant topics. These example lists contain correct answers but are not exhaustive. Use them as guidance, but also make your own final judgment about what qualifies as correct when appropriate.
"""

BINARY_USER_PROMPT = """## Document Content
{document_content}

## Rubric Criterion to Evaluate

**Title**: {rubric_title}
**Category**: {rubric_category}
**Weight**: {rubric_weight}

Important: Judge whether the criterion itself is present in the document. Some criteria describe undesirable behavior
and have a negative weight. Do not invert the binary mapping for those criteria: if the undesirable behavior is
present, return Satisfied with score 1.0; if it is absent, return Not Satisfied with score 0.0.

## Your Task

Evaluate whether the above document satisfies this specific rubric criterion.

## Required Response Format

Provide your evaluation in the following JSON format:

```json
{{
  "verdict": "[Not Satisfied/Satisfied]",
  "score": [0.0/1.0],
  "confidence": [0.0-1.0],
  "reasoning": "Detailed explanation with specific evidence from the document",
  "evidence_quotes": ["Direct quote 1", "Direct quote 2"],
  "missing_elements": ["Element 1 that would improve satisfaction"]
}}
```

Ensure your response is ONLY the JSON object, with no additional text.
"""

CHUNK_SYSTEM_PROMPT = 'You are evaluating document chunks for rubric criteria.'

CHUNK_USER_PROMPT = """You are evaluating a large document in chunks. This is chunk {chunk_num} of {total_chunks}.

## Previous Context Summary
{context_summary}

## Current Chunk Content
{chunk_content}

## Rubric Criterion
**Title**: {rubric_title}
**Category**: {rubric_category}

Please evaluate this chunk for evidence related to the rubric criterion. Your response should be in JSON format:

```json
{{
  "relevant_evidence": ["Evidence point 1", "Evidence point 2"],
  "satisfaction": true/false,
  "confidence_for_chunk": [0.0-1.0],
  "notes": "Any important observations"
}}
```
"""

SYNTHESIS_USER_PROMPT = """Based on the following evidence collected from the document:

Evidence points:
{all_evidence}

Evaluate whether the document satisfies the rubric criterion:
**Title**: {rubric_title}
**Category**: {rubric_category}

Provide your final evaluation in JSON format:
{{
  "verdict": "[Not Satisfied/Satisfied]",
  "score": [0.0/1.0],
  "confidence": [0.0-1.0],
  "reasoning": "Synthesis of evidence",
  "evidence_quotes": ["Evidence point 1"],
  "missing_elements": ["Missing element 1"]
}}
"""


class TemporaryLocalAgentEnvironment(LocalAgentEnvironment):
    """Local agent environment with an isolated temporary working directory."""

    def __init__(self, sample_id: Any) -> None:
        safe_id = ''.join(char if str(char).isalnum() else '-' for char in str(sample_id))
        self._temporary_directory = tempfile.TemporaryDirectory(prefix=f'evalscope-researchrubrics-{safe_id}-')
        super().__init__(working_dir=self._temporary_directory.name)

    @property
    def working_dir(self) -> Path:
        return Path(self._temporary_directory.name)

    async def close(self) -> None:
        await super().close()
        self._temporary_directory.cleanup()


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    data = json.loads(strip_json_fence(text))
    if not isinstance(data, dict):
        raise ValueError('Judge response must be a JSON object.')
    return data


def validate_binary_result(data: Dict[str, Any]) -> Dict[str, Any]:
    verdict = data.get('verdict')
    expected_scores = {'Not Satisfied': 0.0, 'Satisfied': 1.0}
    if verdict not in expected_scores:
        raise ValueError(f'Invalid binary verdict: {verdict!r}.')
    try:
        score = float(data.get('score'))
        confidence = float(data.get('confidence'))
    except (TypeError, ValueError) as exc:
        raise ValueError('Judge score and confidence must be numeric.') from exc
    if score != expected_scores[verdict]:
        raise ValueError(f'Judge verdict {verdict!r} conflicts with score {score}.')
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f'Judge confidence must be between 0 and 1, got {confidence}.')
    reasoning = data.get('reasoning')
    evidence_quotes = data.get('evidence_quotes')
    missing_elements = data.get('missing_elements')
    if not isinstance(reasoning, str):
        raise ValueError('Judge reasoning must be a string.')
    if not isinstance(evidence_quotes, list) or not all(isinstance(item, str) for item in evidence_quotes):
        raise ValueError('Judge evidence_quotes must be a list of strings.')
    if not isinstance(missing_elements, list) or not all(isinstance(item, str) for item in missing_elements):
        raise ValueError('Judge missing_elements must be a list of strings.')
    return {
        'verdict': verdict,
        'score': score,
        'confidence': confidence,
        'reasoning': reasoning,
        'evidence_quotes': evidence_quotes,
        'missing_elements': missing_elements,
    }


def validate_chunk_result(data: Dict[str, Any]) -> Dict[str, Any]:
    evidence = data.get('relevant_evidence')
    satisfaction = data.get('satisfaction')
    confidence = data.get('confidence_for_chunk')
    notes = data.get('notes')
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ValueError('Chunk relevant_evidence must be a list of strings.')
    if not isinstance(satisfaction, bool):
        raise ValueError('Chunk satisfaction must be a boolean.')
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError('Chunk confidence_for_chunk must be numeric.') from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f'Chunk confidence must be between 0 and 1, got {confidence}.')
    if not isinstance(notes, str):
        raise ValueError('Chunk notes must be a string.')
    return {
        'relevant_evidence': evidence,
        'satisfaction': satisfaction,
        'confidence_for_chunk': confidence,
        'notes': notes,
    }


def chunk_document(content: str, max_tokens: int) -> List[str]:
    max_chars = max_tokens * 4
    if len(content) <= max_chars:
        return [content]

    chunks: List[str] = []
    current: List[str] = []
    current_length = 0
    for paragraph in content.split('\n\n'):
        paragraph_length = len(paragraph) + 2
        if current and current_length + paragraph_length > max_chars:
            chunks.append('\n\n'.join(current).strip())
            current = []
            current_length = 0
        if paragraph_length > max_chars:
            for start in range(0, len(paragraph), max_chars):
                if current:
                    chunks.append('\n\n'.join(current).strip())
                    current = []
                    current_length = 0
                chunks.append(paragraph[start:start + max_chars])
            continue
        current.append(paragraph)
        current_length += paragraph_length
    if current:
        chunks.append('\n\n'.join(current).strip())
    return chunks
