# Copyright (c) Alibaba, Inc. and its affiliates.

import inspect
import re
import signal
from collections import defaultdict
from tqdm import tqdm


def handle(signum, frame):
    raise RuntimeError('程序执行超过10秒')


def check_input(text, arg):
    pattern = r'input\((.*?)\n'
    text = re.sub(pattern, '{}\n'.format(arg), text)

    code_block_pattern = re.compile(r'```[Pp]ython\n(.*?)\n```', re.DOTALL)
    code_block = code_block_pattern.search(text)
    code_string = code_block.group(1)

    function_name_pattern = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\(', re.DOTALL)
    function_name_block = function_name_pattern.search(code_string)
    function_name = function_name_block.group(1)

    return code_string, function_name


def compile_func(code_string, function_name):
    signal.signal(signal.SIGALRM, handle)
    signal.alarm(10)
    myMod = compile(code_string, '', 'exec')
    exec(myMod)
    func = eval(function_name)
    signal.alarm(0)
    return func


def exec_func(func, arr):
    signal.signal(signal.SIGALRM, handle)
    signal.alarm(10)
    sig = inspect.signature(func)
    params = [param for param in sig.parameters]
    if len(params) == 0:
        result = func()
    else:
        result = func(arr)
    signal.alarm(0)
    return result


def compute_pass_k_one_sample(predict, func_args, func_outputs, k=4):
    assert len(predict) >= k, f'pass@k must have {k} generations, now have {len(predict)}'
    for predict_i in predict[:k]:
        try:
            for arg, gold in zip(func_args, func_outputs):
                code_string, function_name = check_input(predict_i, arg)
                func = compile_func(code_string, function_name)

                exec_result = exec_func(func, arg)
                if type(exec_result) is tuple:
                    exec_result = list(exec_result)
                exec_result = str(exec_result).lower()
                assert exec_result == str(gold).lower()
                del func
            return 1
        except Exception as e:
            print(e)
    return 0


def compute_pass_k(predict_l, reference_l, func_args_l, k=4, lang='py'):
    if lang != 'py':
        print('only support python code.')

    assert len(predict_l) == len(reference_l) == len(func_args_l)
    pass_k_cnt = 0
    for predict, ref, func_args in zip(predict_l, reference_l, func_args_l):
        pass_k_cnt += compute_pass_k_one_sample(predict, ref, func_args, k)
    return {'pass@k': pass_k_cnt / len(predict_l)}


def run_code_eval(data_l, k=4, md_level=2):
    print(f"{'#' * md_level} Code Eval(pass@{k})")
    for data in tqdm(data_l):
        data[f'pass@{k}'] = compute_pass_k_one_sample(data['gen'], data['func_args'], data['func_outputs'], k)
    task_data_d = defaultdict(list)
    for data in data_l:
        for task in data['task_tags']:
            task_data_d[task].append(data)

    correct_cnt = sum([data[f'pass@{k}'] for data in data_l])
    print(f'[total], count: {len(data_l)}, pass@{k}: '
          f'{correct_cnt / len(data_l) * 100:0.2f}%')
    for task in task_data_d.keys():
        correct_cnt = sum([data[f'pass@{k}'] for data in task_data_d[task]])
        print(f'[{task}], count: {len(task_data_d[task])}, pass@{k}: '
              f'{correct_cnt / len(task_data_d[task]) * 100:0.2f}%')
