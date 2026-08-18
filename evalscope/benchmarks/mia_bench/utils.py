# flake8: noqa: E501
from typing import Dict, List

from evalscope.utils.logger import get_logger

logger = get_logger()


def _build_components_text(components: List[str]) -> str:
    """Build the component description text for the judge prompt."""
    n = len(components)
    ordinals = ['first', 'second', 'third', 'fourth', 'fifth']
    if n == 1:
        return f"The first component is:' {components[0]}'"
    parts = [f"the {ordinals[i]} component is:' {components[i]}'" for i in range(n)]
    # Capitalise first word
    parts[0] = 'T' + parts[0][1:]
    return ', and '.join(parts)


def _build_score_text(components: List[str], weights: List[int]) -> str:
    """Build the score description text for the judge prompt."""
    n = len(components)
    w = [str(wt) for wt in weights]
    if n == 1:
        return f'The first component is worth {w[0]} scores.'
    elif n == 2:
        return f'The first and second component is each worth {w[0]} and {w[1]} scores.'
    elif n == 3:
        return f'The first second, and third component is each worth {w[0]}, {w[1]} and {w[2]} scores.'
    elif n == 4:
        return f'The first second, third, and fourth component is each worth {w[0]}, {w[1]}, {w[2]} and {w[3]} scores.'
    elif n == 5:
        return f'The first second, third, fourth and fifth component is each worth {w[0]}, {w[1]}, {w[2]}, {w[3]} and {w[4]} scores.'
    else:
        ordinals = ['first', 'second', 'third', 'fourth', 'fifth']
        pieces = [f'{ordinals[i]} worth {w[i]}' for i in range(n)]
        return 'Components are each worth: ' + ', '.join(pieces) + ' scores.'


def generate_mia_judge_prompt(
    instruction: str,
    components: List[str],
    component_weight: List[int],
    response: str,
) -> str:
    """
    Generate the LLM judge evaluation prompt for MIA-Bench.

    Equivalent to reference.py generate_prompt() but decoupled from the doc dict. The reply format
    is declared by the caller's OutputContract; ``component_i`` maps to the i-th component here.
    """
    components_text = _build_components_text(components)
    score_text = _build_score_text(components, component_weight)
    total = sum(component_weight)

    prompt = (
        f"Here is an instruction for a multimodal LLM: ' {instruction}"
        f' You need to grade if the response from the model follows each component of the instruction. '
        f'{components_text}'
        f" The response is:' {response}'"
        f' You need to score the response and be strict. The total score ranges from 0 to {total}, depending on if the response follows the instruction. '
        f'{score_text}'
    )
    return prompt
