import json

easy_sizes = [
    '2*2',
    '2*3',
    '2*4',
    '2*5',
    '2*6',
    '3*2',
    '3*3',
]
hard_sizes = [
    '3*4', '3*5', '4*2', '3*6', '4*3', '4*4', '5*2', '6*2', '4*5', '4*6', '5*3', '5*4', '5*5', '5*6', '6*3', '6*4',
    '6*5', '6*6'
]

small_sizes = ['2*2', '2*3', '2*4', '2*5', '2*6', '3*2', '3*3', '4*2']
medium_sizes = ['3*4', '3*5', '3*6', '4*3', '4*4', '5*2', '6*2']
large_sizes = ['4*5', '5*3', '4*6', '5*4', '6*3']
xl_sizes = ['5*5', '6*4', '5*6', '6*5', '6*6']


def process_results(prediction, reference):
    solved_puzzles = 0
    correct_cells = 0
    total_cells = 0
    no_answer = 0
    num_total_puzzles_by_size = {}
    solved_puzzles_by_size = {}
    reason_lens = []

    prediction = json.loads(prediction)
    reference = json.loads(reference)

    num_houses = len(reference['rows'])
    num_person = len(reference['rows'][0]) - 1
    size = f'{num_houses}*{num_person}'
    if size not in num_total_puzzles_by_size:
        num_total_puzzles_by_size[size] = 0
    num_total_puzzles_by_size[size] += 1
    if size not in solved_puzzles_by_size:
        solved_puzzles_by_size[size] = 0

    columns = reference['header']

    reference_table = {}
    num_houses = len(reference['rows'])
    this_total_cells = 0
    for i in range(num_houses):
        reference_table[f'House {i + 1}'] = {columns[j]: reference['rows'][i][j] for j in range(1, len(columns))}
        this_total_cells += len(columns) - 1
    total_cells += this_total_cells

    prediction_table = prediction.get('solution', None)

    if prediction_table:
        reason = prediction.get('reasoning', '')
        reason_lens.append(len(reason))

        this_correct_cells = 0
        for house in reference_table:
            for column in reference_table[house]:
                if house in prediction_table and column in prediction_table[house]:
                    truth_cell = reference_table[house][column].lower().strip()
                    if prediction_table[house][column] is None or len(prediction_table[house][column]) == 0:
                        continue
                    if isinstance(prediction_table[house][column], list):
                        predicted_cell = prediction_table[house][column][0].lower().strip()
                    elif isinstance(prediction_table[house][column], str):
                        predicted_cell = prediction_table[house][column].lower().strip()
                    else:
                        raise ValueError(f'Unknown type: {type(prediction_table[house][column])}')
                    if truth_cell.lower().strip() == predicted_cell.lower().strip():
                        this_correct_cells += 1
        correct_cells += this_correct_cells

        if this_correct_cells == this_total_cells:
            solved_puzzles += 1
            solved_puzzles_by_size[size] += 1
    else:
        no_answer += 1

    easy_solved_puzzles = sum([
        solved_puzzles_by_size[size] if size in solved_puzzles_by_size else 0 for size in easy_sizes
    ])
    easy_total_puzzles = sum([
        num_total_puzzles_by_size[size] if size in num_total_puzzles_by_size else 0 for size in easy_sizes
    ])
    hard_solved_puzzles = sum([
        solved_puzzles_by_size[size] if size in solved_puzzles_by_size else 0 for size in hard_sizes
    ])
    hard_total_puzzles = sum([
        num_total_puzzles_by_size[size] if size in num_total_puzzles_by_size else 0 for size in hard_sizes
    ])

    small_solved_puzzles = sum([
        solved_puzzles_by_size[size] if size in solved_puzzles_by_size else 0 for size in small_sizes
    ])
    small_total_puzzles = sum([
        num_total_puzzles_by_size[size] if size in num_total_puzzles_by_size else 0 for size in small_sizes
    ])
    medium_solved_puzzles = sum([
        solved_puzzles_by_size[size] if size in solved_puzzles_by_size else 0 for size in medium_sizes
    ])
    medium_total_puzzles = sum([
        num_total_puzzles_by_size[size] if size in num_total_puzzles_by_size else 0 for size in medium_sizes
    ])
    large_solved_puzzles = sum([
        solved_puzzles_by_size[size] if size in solved_puzzles_by_size else 0 for size in large_sizes
    ])
    large_total_puzzles = sum([
        num_total_puzzles_by_size[size] if size in num_total_puzzles_by_size else 0 for size in large_sizes
    ])
    xl_solved_puzzles = sum([
        solved_puzzles_by_size[size] if size in solved_puzzles_by_size else 0 for size in xl_sizes
    ])
    xl_total_puzzles = sum([
        num_total_puzzles_by_size[size] if size in num_total_puzzles_by_size else 0 for size in xl_sizes
    ])

    result = {}

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
