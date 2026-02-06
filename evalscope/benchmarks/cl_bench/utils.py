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
            lines.append(f"{index}. {criteria}")
    return '\n'.join(lines) if lines else 'No specific rubrics provided.'


def extract_json_block(result_text: str) -> Optional[str]:
    # Remove code block wrapper if present
    if result_text.startswith('```json'):
        result_text = result_text[7:]
    if result_text.startswith('```'):
        result_text = result_text[3:]
    if result_text.endswith('```'):
        result_text = result_text[:-3]
    result_text = result_text.strip()

    return result_text
