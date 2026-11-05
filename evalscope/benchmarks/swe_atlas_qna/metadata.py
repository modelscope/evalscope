from __future__ import annotations

from typing import Any, Dict

DATASET_ID = 'ScaleAI/SWE-Atlas-QnA'
HF_DATASET_ID = 'ScaleAI/SWE-Atlas-QnA'
WORKING_DIR = '/app'

EXTRA_PARAMS: Dict[str, Any] = {
    'working_dir': {
        'type': 'str',
        'description': 'Repository path inside each task image.',
        'value': WORKING_DIR,
    },
    'reset_to_base_commit': {
        'type': 'bool',
        'description': 'Reset the repository to repository_base_commit before the agent starts.',
        'value': True,
    },
    'fail_on_source_modification': {
        'type': 'bool',
        'description': 'Fail tasks whose repository contains uncommitted changes after the agent run.',
        'value': True,
    },
    'command_timeout': {
        'type': 'float',
        'description': 'Timeout in seconds for each sandbox command when no native timeout is configured.',
        'value': 900.0,
    },
    'pass_threshold': {
        'type': 'float',
        'description': 'Minimum rubric pass rate required to resolve a task.',
        'value': 1.0,
    },
}

DESCRIPTION = """
## Overview

SWE-Atlas QnA is a codebase question-answering benchmark from Scale AI. The public set contains 124 read-only
code-comprehension tasks over 11 production repositories in Go, Python, C, and TypeScript.

## Task Description

- **Task Type**: Multi-turn codebase exploration and question answering
- **Input**: A natural-language question and a task-specific repository image
- **Output**: A prose answer supported by evidence from the repository
- **Domain**: Software engineering and code comprehension

## Key Features

- Runs EvalScope's native AgentLoop with a bash tool inside each task's Docker image.
- Resets the repository to its pinned base commit before evaluation and rejects answers accompanied by source edits.
- Supports positive and negative expert-written rubric items.
- Reports task resolve rate and mean rubric pass rate.

## Evaluation Notes

- `resolve_rate` is the primary metric; a task is resolved when its rubric pass rate reaches `pass_threshold`, which
  defaults to 1.0, and the repository remains clean.
- Each rubric item is graded independently by an LLM judge using the official output schema. Invalid judge replies,
  repository setup failures, and source-tree check failures exclude the sample rather than assigning a zero score.
- A real run requires Docker, `ms-enclave`, and the task images referenced by the dataset. Images can require
  approximately 20-35 GB of disk space in total.
- This adapter uses EvalScope's AgentLoop rather than the official SWE-agent and Modal scaffold, so results are not
  claimed to be leaderboard-equivalent without a separate comparison.
""".strip()
