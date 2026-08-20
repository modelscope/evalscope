from typing import Any, Iterable, Optional

# align with official CL-bench eval.py(https://github.com/Tencent-Hunyuan/CL-bench/blob/main/eval.py)


def build_rubrics_text(rubrics: Optional[Iterable[Any]]) -> str:
    if not rubrics:
        return 'No specific rubrics provided.'

    lines = []
    for index, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = str(rubric.get('rubric_criteria', '')).strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f'{index}. {criteria}')
    return '\n'.join(lines) if lines else 'No specific rubrics provided.'
