# Scoring for FinQA, ported from the official evaluation in https://github.com/czyssrs/FinQA
# (code/evaluate/evaluate.py, MIT License, Copyright (c) 2021 Zhiyu Chen).
#
# A prediction is a program in the FinQA DSL, e.g. ``subtract(5829, 5735), divide(#0, 5735)``. Execution
# accuracy runs it on the question's table and compares the result with the gold answer exactly (both are
# rounded to 5 decimals); program accuracy checks that it is symbolically equal to the gold program.

import re
from typing import List, Optional, Tuple, Union

ALL_OPS = [
    'add',
    'subtract',
    'multiply',
    'divide',
    'exp',
    'greater',
    'table_max',
    'table_min',
    'table_sum',
    'table_average',
]

_PROGRAM_RE = re.compile(r'^\s*\**\s*program\s*\**\s*:\s*(.*)$', re.IGNORECASE | re.MULTILINE)

Number = Union[float, str]


def str_to_num(text: str) -> Number:
    text = text.replace(',', '')
    try:
        num = float(text)
    except ValueError:
        if '%' in text:
            text = text.replace('%', '')
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = 'n/a'
        elif 'const' in text:
            text = text.replace('const_', '')
            if text == 'm1':
                text = '-1'
            num = float(text)
        else:
            num = 'n/a'
    return num


def process_row(row_in: List[str]):
    row_out = []
    for num in row_in:
        num = num.replace('$', '').strip()
        num = num.split('(')[0].strip()
        num = str_to_num(num)
        if num == 'n/a':
            return 'n/a'
        row_out.append(num)
    return row_out


def eval_program(program: List[str], table: List[List[str]]) -> Tuple[int, Number]:
    """Execute a tokenized program; returns ``(invalid_flag, result)``."""
    invalid_flag = 0
    this_res = 'n/a'
    try:
        program = program[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip('(') not in ALL_OPS:
                    return 1, 'n/a'
            if (ind + 1) % 4 == 0:
                if token != ')':
                    return 1, 'n/a'

        program = '|'.join(program)
        steps = program.split(')')[:-1]
        res_dict = {}
        for ind, step in enumerate(steps):
            step = step.strip()
            if len(step.split('(')) > 2:
                invalid_flag = 1
                break
            op = step.split('(')[0].strip('|').strip()
            args = step.split('(')[1].strip('|').strip()
            arg1 = args.split('|')[0].strip()
            arg2 = args.split('|')[1].strip()

            if op in ('add', 'subtract', 'multiply', 'divide', 'exp', 'greater'):
                if '#' in arg1:
                    arg1 = res_dict[int(arg1.replace('#', ''))]
                else:
                    arg1 = str_to_num(arg1)
                    if arg1 == 'n/a':
                        invalid_flag = 1
                        break
                if '#' in arg2:
                    arg2 = res_dict[int(arg2.replace('#', ''))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == 'n/a':
                        invalid_flag = 1
                        break

                if op == 'add':
                    this_res = arg1 + arg2
                elif op == 'subtract':
                    this_res = arg1 - arg2
                elif op == 'multiply':
                    this_res = arg1 * arg2
                elif op == 'divide':
                    this_res = arg1 / arg2
                elif op == 'exp':
                    this_res = arg1**arg2
                elif op == 'greater':
                    this_res = 'yes' if arg1 > arg2 else 'no'
                res_dict[ind] = this_res

            elif 'table' in op:
                table_dict = {}
                for row in table:
                    table_dict[row[0]] = row[1:]
                if '#' in arg1:
                    arg1 = res_dict[int(arg1.replace('#', ''))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    cal_row = table_dict[arg1]
                    num_row = process_row(cal_row)
                if num_row == 'n/a':
                    invalid_flag = 1
                    break
                if op == 'table_max':
                    this_res = max(num_row)
                elif op == 'table_min':
                    this_res = min(num_row)
                elif op == 'table_sum':
                    this_res = sum(num_row)
                elif op == 'table_average':
                    this_res = sum(num_row) / len(num_row)
                res_dict[ind] = this_res

        if this_res != 'yes' and this_res != 'no' and this_res != 'n/a':
            this_res = round(this_res, 5)
    except Exception:
        invalid_flag = 1

    return invalid_flag, this_res


def equal_program(program1: List[str], program2: List[str]) -> bool:
    """Whether the predicted program (``program2``) is symbolically equal to the gold one (``program1``)."""
    from sympy import simplify

    sym_map = {}

    program1 = program1[:-1]  # remove EOF
    program1 = '|'.join(program1)
    steps = program1.split(')')[:-1]

    sym_ind = 0
    step_dict_1 = {}

    # symbolic map
    for ind, step in enumerate(steps):
        step = step.strip()
        assert len(step.split('(')) <= 2
        op = step.split('(')[0].strip('|').strip()
        args = step.split('(')[1].strip('|').strip()
        arg1 = args.split('|')[0].strip()
        arg2 = args.split('|')[1].strip()
        step_dict_1[ind] = step

        if 'table' in op:
            if step not in sym_map:
                sym_map[step] = 'a' + str(sym_ind)
                sym_ind += 1
        else:
            if '#' not in arg1:
                if arg1 not in sym_map:
                    sym_map[arg1] = 'a' + str(sym_ind)
                    sym_ind += 1
            if '#' not in arg2:
                if arg2 not in sym_map:
                    sym_map[arg2] = 'a' + str(sym_ind)
                    sym_ind += 1

    # check program 2
    step_dict_2 = {}
    try:
        program2 = program2[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program2):
            if ind % 4 == 0:
                if token.strip('(') not in ALL_OPS:
                    return False
            if (ind + 1) % 4 == 0:
                if token != ')':
                    return False

        program2 = '|'.join(program2)
        steps = program2.split(')')[:-1]

        for ind, step in enumerate(steps):
            step = step.strip()
            if len(step.split('(')) > 2:
                return False
            op = step.split('(')[0].strip('|').strip()
            args = step.split('(')[1].strip('|').strip()
            arg1 = args.split('|')[0].strip()
            arg2 = args.split('|')[1].strip()
            step_dict_2[ind] = step

            if 'table' in op:
                if step not in sym_map:
                    return False
            else:
                if '#' not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    if int(arg1.strip('#')) >= ind:
                        return False
                if '#' not in arg2:
                    if arg2 not in sym_map:
                        return False
                else:
                    if int(arg2.strip('#')) >= ind:
                        return False
    except Exception:
        return False

    def symbol_recur(step, step_dict):
        step = step.strip()
        op = step.split('(')[0].strip('|').strip()
        args = step.split('(')[1].strip('|').strip()
        arg1 = args.split('|')[0].strip()
        arg2 = args.split('|')[1].strip()

        if 'table' in op:
            # as var
            return sym_map[step]

        if '#' in arg1:
            arg1_part = symbol_recur(step_dict[int(arg1.replace('#', ''))], step_dict)
        else:
            arg1_part = sym_map[arg1]

        if '#' in arg2:
            arg2_part = symbol_recur(step_dict[int(arg2.replace('#', ''))], step_dict)
        else:
            arg2_part = sym_map[arg2]

        if op == 'add':
            return '( ' + arg1_part + ' + ' + arg2_part + ' )'
        elif op == 'subtract':
            return '( ' + arg1_part + ' - ' + arg2_part + ' )'
        elif op == 'multiply':
            return '( ' + arg1_part + ' * ' + arg2_part + ' )'
        elif op == 'divide':
            return '( ' + arg1_part + ' / ' + arg2_part + ' )'
        elif op == 'exp':
            return '( ' + arg1_part + ' ** ' + arg2_part + ' )'
        elif op == 'greater':
            return '( ' + arg1_part + ' > ' + arg2_part + ' )'

    # derive symbolic program 1
    steps = program1.split(')')[:-1]
    sym_prog1 = symbol_recur(steps[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)

    try:
        # derive symbolic program 2
        steps = program2.split(')')[:-1]
        sym_prog2 = symbol_recur(steps[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
    except Exception:
        return False

    return sym_prog1 == sym_prog2


def program_tokenization(original_program: str) -> List[str]:
    """``subtract(5829, 5735)`` -> ``['subtract(', '5829', '5735', ')', 'EOF']``."""
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


# --- model output parsing ---------------------------------------------------------------------------------


def parse_program(text: str) -> Optional[str]:
    """The program on the last ``Program:`` line, with whitespace around brackets and separators normalized.

    Steps and arguments are separated by a comma and a space, as in the dataset. A step whose two arguments
    are joined by a bare comma, e.g. ``divide(6.3,18.1)``, gets the space added.
    """
    found = _PROGRAM_RE.findall(text or '')
    if not found:
        return None
    program = found[-1].strip().strip('*').strip().rstrip('.').strip().strip('`').strip()
    program = re.sub(r'\(\s+', '(', program)
    program = re.sub(r'\s+\)', ')', program)
    program = re.sub(r'\s+,', ',', program)
    program = re.sub(r',\s+', ', ', program)
    # `op(a,b)`: a single comma without a space between two arguments is the separator
    program = re.sub(r'\(([^(),\s][^(),]*),([^(),\s][^(),]*)\)', r'(\1, \2)', program)
    return program or None


def score_program(
    program: Optional[str], gold_program: str, gold_answer: Number, table: List[List[str]]
) -> Tuple[float, float]:
    """Execution accuracy and program accuracy of one prediction, as in the official ``evaluate_result``."""
    if not program:
        return 0.0, 0.0
    pred = program_tokenization(program)
    gold = program_tokenization(gold_program)
    invalid_flag, exe_res = eval_program(pred, table)
    exe_correct = float(invalid_flag == 0 and exe_res == gold_answer)
    prog_correct = float(equal_program(gold, pred))
    return exe_correct, prog_correct
