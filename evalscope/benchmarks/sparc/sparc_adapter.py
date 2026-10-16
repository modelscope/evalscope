# SPaRC adapter: 2D grid pathfinding puzzles with rule constraints (EMNLP 2025).
#
# Paper: https://aclanthology.org/2025.emnlp-main.526/ (arXiv: https://arxiv.org/abs/2505.16686)
# Dataset: https://www.modelscope.cn/datasets/evalscope/SPaRC (CC BY-SA 4.0)
# Reference implementation: https://github.com/lkaesberg/SPaRC (MIT)

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from .utils import SYSTEM_PROMPT, analyze_path, extract_solution_path, get_prompt, path_to_text, validate_solution

logger = get_logger()

# Pinned commit of the ModelScope dataset repository (2026-10).
DATASET_REVISION = '104103a0abae909a46b85f11fb2f949fc7448f87'

SUBSET_LIST = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

# Fields the prompt builder and the validator read from a puzzle record.
PUZZLE_FIELDS = ('id', 'grid_size', 'puzzle_array', 'polyshapes', 'solutions')

DESCRIPTION = """
## Overview

SPaRC (Spatial Pathfinding Reasoning Challenge) tests step-by-step spatial reasoning with 2D grid
puzzles based on the game *The Witness*. The model must draw one line from a start node to an end
node that obeys every rule on the grid.

## Task Description

- **Task Type**: Spatial pathfinding under rule constraints
- **Input**: A text grid (start and end nodes, gaps, dots, colored squares, stars, triangles,
  polyominoes) and the rules of the game
- **Output**: Step-by-step reasoning, then the marker `####` and the path as a list of `(x, y)`
  node coordinates
- **Domain**: Spatial reasoning, constraint satisfaction

## Key Features

- 500 test puzzles in five difficulty levels (Level 1 to Level 5: 86 / 118 / 121 / 86 / 89 puzzles);
  the paper calls Level 5 "hard"
- Every valid solution of each puzzle is stored in the dataset, so scoring needs no judge model
- In the paper, humans solve 98.0% of the test puzzles (94.5% of Level 5), and the best model,
  o4-mini, solves 15.8% (1.1% of Level 5)
- The prompt is the paper's single-shot prompt, with one worked example inside the prompt

## Evaluation Notes

- Subsets are the difficulty levels; `subset_list: ['Level 5']` evaluates only the hard puzzles.
  `limit` applies per level, so `limit=4` gives 20 puzzles (4 per level)
- `accuracy` (primary): the path after the last `####` must exactly match one of the puzzle's
  stored solutions
- `invalid_path_rate` (lower is better): share of answers that are not a valid line from start to
  end (wrong start or end, a gap between steps, a revisited node, or a step on a rule cell). This
  is the "Invalid Path" column of Table 7 in the paper
- The prompt builder and the validator are copies of the reference code in the `sparc-puzzle`
  package; they give the same prompts and the same scores on all 500 test puzzles
- The dataset is `evalscope/SPaRC` on ModelScope (config `all`), with the revision pinned for
  reproducibility
- Reasoning models often write more than 10k tokens per puzzle, so set a large `max_tokens`
- [Paper](https://aclanthology.org/2025.emnlp-main.526/) | [Code](https://github.com/lkaesberg/SPaRC) |
  [Dataset](https://www.modelscope.cn/datasets/evalscope/SPaRC) | [Website](https://sparc.gipplab.org/)
"""


@register_benchmark(
    BenchmarkMeta(
        name='sparc',
        pretty_name='SPaRC',
        dataset_id='evalscope/SPaRC',
        dataset_revision=DATASET_REVISION,
        hf_dataset_id='lkaesberg/SPaRC',
        hf_dataset_revision='82de88e9ed0594ab6694beb2436db272daba39ea',
        paper_url='https://arxiv.org/abs/2505.16686',
        tags=[Tags.REASONING],
        description=DESCRIPTION,
        subset_list=SUBSET_LIST,
        default_subset='all',
        eval_split='test',
        few_shot_mode='disabled',
        metric_list=['accuracy', 'invalid_path_rate'],
        primary_metric='accuracy',
        system_prompt=SYSTEM_PROMPT,
        prompt_template='{question}',
        evaluation_version='v1.0',
    )
)
class SPaRCAdapter(DefaultDataAdapter):
    """Data adapter for SPaRC: exact match of the extracted path against the stored solutions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Load the single `all` config once and split it into one subset per difficulty level.
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Build the paper's single-shot prompt and keep the puzzle for scoring."""
        return Sample(
            input=get_prompt(record),
            target=path_to_text(record['solutions'][0]['path']),
            subset_key=f'Level {record["difficulty_level"]}',
            metadata={
                'difficulty_level': record['difficulty_level'],
                'difficulty_score': record['difficulty_score'],
                'solution_count': record['solution_count'],
                'puzzle': {key: record[key] for key in PUZZLE_FIELDS},
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Return the path the reference parser reads from the answer, as ``(x,y), ...`` text."""
        path = extract_solution_path(prediction, task_state.metadata['puzzle'])
        return path_to_text(path)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score with the reference validator: exact match and the structural invalid-path check."""
        puzzle = task_state.metadata['puzzle']
        # Parsing the extracted text again gives the same path as parsing the full answer.
        path = extract_solution_path(filtered_prediction, puzzle)
        solved = validate_solution(path, puzzle)
        analysis = analyze_path(path, puzzle)
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                'accuracy': float(solved),
                'invalid_path_rate': float(not analysis['fully_valid_path']),
            },
            main_score_name='accuracy',
            metadata={'path_analysis': analysis},
        )
