def get_single_answer_type_text(answer_type, is_chinese):
    if '-' in answer_type:  # No need now
        answer_type = answer_type[: answer_type.find('-')]
    chinese_answer_type_dict = {
        'Numerical': '数值',
        'Expression': '表达式',
        'Equation': '方程',
        'Interval': '区间',
    }
    english_answer_type_dict = {
        'Numerical': 'a numerical value',
        'Expression': 'an expression',
        'Equation': 'an equation',
        'Interval': 'an interval',
    }

    for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
        if t in answer_type:
            if is_chinese:
                return chinese_answer_type_dict[t]
            else:
                return english_answer_type_dict[t]
    raise ValueError(f'Error parsing answer type {answer_type}!')


def get_answer_type_text(answer_type, is_chinese, multiple_answer):
    if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):
        return ''
    if not multiple_answer:
        answer_text = get_single_answer_type_text(answer_type, is_chinese)
        if is_chinese:
            return f'，答案类型为{answer_text}'
        else:
            return f'The answer of The problem should be {answer_text}. '
    # Multiple answers case
    if ',' not in answer_type:  # Same answer type for all answers
        answer_text = get_single_answer_type_text(answer_type, is_chinese)
        if is_chinese:
            return f'，题目有多个答案，答案类型均为{answer_text}'
        else:
            return f'The problem has multiple answers, each of them should be {answer_text}. '
    # Different answer types
    answer_types = answer_type.split(',')
    answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
    if len(set(answer_types)) == 1:
        answer_text = answer_types[0]
        if is_chinese:
            return f'，题目有多个答案，答案类型均为{answer_text}'
        else:
            return f'The problem has multiple answers, each of them should be {answer_text}. '
    else:
        if is_chinese:
            answer_text = '、'.join(answer_types)
            return f'，题目有多个答案，答案类型分别为{answer_text}'
        else:
            answer_text = ', '.join(answer_types)
            return f'The problem has multiple answers, with the answers in order being {answer_text}. '


class OlympiadBenchPrompter:
    def __init__(self):
        pass

    def make_prompt(
        self,
        problem,
        language,
        subject,
        question_type,
        answer_type,
        is_multiple_answer,
        unit,
    ):
        self.is_chinese = language == 'Chinese'
        self.is_math = subject == 'Math'
        self.is_theorem_proving = question_type == 'Theorem proof'
        """Generate prompt based on question properties."""
        if self.is_chinese:
            subject_content = '数学' if self.is_math else '物理'
            if self.is_theorem_proving:
                prompt = (
                    f'以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，'
                    f'运用逻辑推理及常用定理证明题目中的命题。证明过程中使用的变量和公式请使用LaTeX格式表示。'
                )
            else:
                answer_type_text = get_answer_type_text(
                    answer_type,
                    is_chinese=True,
                    multiple_answer=is_multiple_answer,
                )
                if is_multiple_answer:
                    multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
                else:
                    multiple_answer_text = '\\boxed{答案}'
                unit_text = ''
                if unit:
                    multiple_answer_text += '(单位)'
                    unit_text = '，注意答案的单位不要放在\\boxed{}中'
                prompt = (
                    f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。'
                    f'请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的'
                    f'变量和公式请使用LaTeX格式表示。请在最后以"所以最终答案是'
                    f'{multiple_answer_text}。"显式给出结果{unit_text}。'
                )
        else:
            subject_content = 'Math' if self.is_math else 'Physics'
            if self.is_theorem_proving:
                prompt = (
                    f'The following is a theorem proving problem from an '
                    f'International {subject_content} competition. Please use '
                    f'logical reasoning and common theorems to prove the '
                    f'proposition in the problem according to the given '
                    f'requirements. Please use LaTeX format to represent the '
                    f'variables and formulas used in the proof.'
                )
            else:
                if is_multiple_answer:
                    multiple_answer_text = '\\boxed{multiple answers connected with commas}'
                else:
                    multiple_answer_text = '\\boxed{answer}'
                unit_text = ''
                if unit:
                    multiple_answer_text += '(unit)'
                    unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
                answer_type_text = get_answer_type_text(
                    answer_type,
                    is_chinese=False,
                    multiple_answer=is_multiple_answer,
                )
                prompt = (
                    f'The following is an open-ended problem from an '
                    f'International {subject_content} competition. '
                    f'{answer_type_text}Please calculate the answer according '
                    f'to the given requirements and the information provided. '
                    f'Please use LaTeX format to represent the variables and '
                    f'formulas used in the solution process and results. '
                    f'Please end your solution with "So the final answer is '
                    f'{multiple_answer_text}." and give the result explicitly'
                    f'{unit_text}.'
                )
        # Add problem statement to the prompt
        prompt = prompt + '\n' + problem + '\n'
        # Add step-by-step reasoning instruction
        if self.is_chinese:
            prompt += '\n请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。'
        else:
            prompt += '\nPlease reason step by step, and put your final answer within \\boxed{}.'
        return prompt
