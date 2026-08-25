import json
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, model_validator

from evalscope.api.judge import OutputContract

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


class BinaryGrade(BaseModel):
    """The binary rubric judge reply — official ResearchRubrics format."""
    verdict: Literal['Not Satisfied', 'Satisfied']
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence_quotes: List[str]
    missing_elements: List[str]

    @model_validator(mode='after')
    def _check_consistency(self) -> 'BinaryGrade':
        expected = 1.0 if self.verdict == 'Satisfied' else 0.0
        if self.score != expected:
            raise ValueError(f'verdict {self.verdict!r} conflicts with score {self.score}')
        return self


class ChunkGrade(BaseModel):
    """Per-chunk evidence extraction reply."""
    relevant_evidence: List[str]
    satisfaction: bool
    confidence_for_chunk: float = Field(ge=0.0, le=1.0)
    notes: str


BINARY_CONTRACT = OutputContract(schema_model=BinaryGrade)
CHUNK_CONTRACT = OutputContract(schema_model=ChunkGrade)


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
