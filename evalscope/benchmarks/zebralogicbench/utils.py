import json
from typing import Dict, List

EASY_SIZES: List[str] = [
    '2*2',
    '2*3',
    '2*4',
    '2*5',
    '2*6',
    '3*2',
    '3*3',
]
HARD_SIZES: List[str] = [
    '3*4', '3*5', '4*2', '3*6', '4*3', '4*4', '5*2', '6*2', '4*5', '4*6', '5*3', '5*4', '5*5', '5*6', '6*3', '6*4',
    '6*5', '6*6'
]

SMALL_SIZES: List[str] = ['2*2', '2*3', '2*4', '2*5', '2*6', '3*2', '3*3', '4*2']
MEDIUM_SIZES: List[str] = ['3*4', '3*5', '3*6', '4*3', '4*4', '5*2', '6*2']
LARGE_SIZES: List[str] = ['4*5', '5*3', '4*6', '5*4', '6*3']
XL_SIZES: List[str] = ['5*5', '6*4', '5*6', '6*5', '6*6']

# Backward-compatible aliases (if imported elsewhere)
easy_sizes = EASY_SIZES
hard_sizes = HARD_SIZES
small_sizes = SMALL_SIZES
medium_sizes = MEDIUM_SIZES
large_sizes = LARGE_SIZES
xl_sizes = XL_SIZES


def process_results(prediction: str, reference: str) -> Dict[str, int]:
    """
    Compare a model prediction against the reference solution for a single puzzle.

    Args:
        prediction: JSON string with fields: {"reasoning": str, "solution": {...}}
        reference: JSON string containing "header" and "rows" defining the ground truth.

    Returns:
        A dictionary with counts for solved puzzles/cells and grouped statistics.
    """
    solved_puzzles = 0
    correct_cells = 0
    total_cells = 0
    no_answer = 0
    num_total_puzzles_by_size: Dict[str, int] = {}
    solved_puzzles_by_size: Dict[str, int] = {}
    reason_lens: List[int] = []

    prediction_obj = json.loads(prediction)
    reference_obj = json.loads(reference)

    num_houses = len(reference_obj['rows'])
    num_person = len(reference_obj['rows'][0]) - 1
    size = f'{num_houses}*{num_person}'

    if size not in num_total_puzzles_by_size:
        num_total_puzzles_by_size[size] = 0
    num_total_puzzles_by_size[size] += 1
    if size not in solved_puzzles_by_size:
        solved_puzzles_by_size[size] = 0

    columns = reference_obj['header']
    # Each row has len(columns) - 1 cells to match
    this_total_cells = (len(columns) - 1) * num_houses
    total_cells += this_total_cells

    # Build a normalized reference table for comparison
    reference_table: Dict[str, Dict[str, str]] = {}
    for i in range(num_houses):
        reference_table[f'House {i + 1}'] = {columns[j]: reference_obj['rows'][i][j] for j in range(1, len(columns))}

    prediction_table = prediction_obj.get('solution', None)

    if prediction_table:
        reason = prediction_obj.get('reasoning', '')
        reason_lens.append(len(reason))

        this_correct_cells = 0
        for house, col_map in reference_table.items():
            for column, truth in col_map.items():
                # Check existence in prediction
                if house not in prediction_table or column not in prediction_table[house]:
                    continue

                truth_cell = truth.lower().strip()
                pred_val = prediction_table[house][column]
                if pred_val is None:
                    continue

                if isinstance(pred_val, list):
                    if not pred_val:
                        continue
                    predicted_cell = str(pred_val[0]).lower().strip()
                elif isinstance(pred_val, str):
                    predicted_cell = pred_val.lower().strip()
                else:
                    # Preserve original behavior to fail fast on unknown types
                    raise ValueError(f'Unknown type: {type(pred_val)}')

                if truth_cell == predicted_cell:
                    this_correct_cells += 1

        correct_cells += this_correct_cells

        if this_correct_cells == this_total_cells:
            solved_puzzles += 1
            solved_puzzles_by_size[size] += 1
    else:
        no_answer += 1

    easy_solved_puzzles = sum([solved_puzzles_by_size[s] if s in solved_puzzles_by_size else 0 for s in EASY_SIZES])
    easy_total_puzzles = sum([
        num_total_puzzles_by_size[s] if s in num_total_puzzles_by_size else 0 for s in EASY_SIZES
    ])
    hard_solved_puzzles = sum([solved_puzzles_by_size[s] if s in solved_puzzles_by_size else 0 for s in HARD_SIZES])
    hard_total_puzzles = sum([
        num_total_puzzles_by_size[s] if s in num_total_puzzles_by_size else 0 for s in HARD_SIZES
    ])

    small_solved_puzzles = sum([solved_puzzles_by_size[s] if s in solved_puzzles_by_size else 0 for s in SMALL_SIZES])
    small_total_puzzles = sum([
        num_total_puzzles_by_size[s] if s in num_total_puzzles_by_size else 0 for s in SMALL_SIZES
    ])
    medium_solved_puzzles = sum([solved_puzzles_by_size[s] if s in solved_puzzles_by_size else 0 for s in MEDIUM_SIZES])
    medium_total_puzzles = sum([
        num_total_puzzles_by_size[s] if s in num_total_puzzles_by_size else 0 for s in MEDIUM_SIZES
    ])
    large_solved_puzzles = sum([solved_puzzles_by_size[s] if s in solved_puzzles_by_size else 0 for s in LARGE_SIZES])
    large_total_puzzles = sum([
        num_total_puzzles_by_size[s] if s in num_total_puzzles_by_size else 0 for s in LARGE_SIZES
    ])
    xl_solved_puzzles = sum([solved_puzzles_by_size[s] if s in solved_puzzles_by_size else 0 for s in XL_SIZES])
    xl_total_puzzles = sum([num_total_puzzles_by_size[s] if s in num_total_puzzles_by_size else 0 for s in XL_SIZES])

    result: Dict[str, int] = {}
    result['Solved Puzzle'] = solved_puzzles
    result['Solved Cell'] = correct_cells
    result['Cell Num'] = total_cells
    result['No answer'] = no_answer
    result['Easy Puzzle Num'] = easy_total_puzzles
    result['Hard Puzzle Num'] = hard_total_puzzles
    result['Small Puzzle Num'] = small_total_puzzles
    result['Medium Puzzle Num'] = medium_total_puzzles
    result['Large Puzzle Num'] = large_total_puzzles
    result['XL Puzzle Num'] = xl_total_puzzles

    result['Solved Easy Puzzle'] = easy_solved_puzzles
    result['Solved Hard Puzzle'] = hard_solved_puzzles
    result['Solved Small Puzzle'] = small_solved_puzzles
    result['Solved Medium Puzzle'] = medium_solved_puzzles
    result['Solved Large Puzzle'] = large_solved_puzzles
    result['Solved XL Puzzle'] = xl_solved_puzzles

    result['Reason Lens'] = sum(reason_lens)
    return result
