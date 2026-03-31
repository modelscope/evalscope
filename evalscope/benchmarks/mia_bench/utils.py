# flake8: noqa: E501
import re
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


def _build_score_format(components: List[str], weights: List[int]) -> str:
    """Build the expected output format string."""
    parts = [f'score of component {i + 1}: x{i + 1}/{weights[i]}' for i in range(len(components))]
    total = sum(weights)
    parts.append(f'total score: z/{total}')
    return ', '.join(parts)


def generate_mia_judge_prompt(
    instruction: str,
    components: List[str],
    component_weight: List[int],
    response: str,
) -> str:
    """
    Generate the LLM judge evaluation prompt for MIA-Bench.

    Equivalent to reference.py generate_prompt() but decoupled from the doc dict.
    """
    components_text = _build_components_text(components)
    score_text = _build_score_text(components, component_weight)
    score_format = _build_score_format(components, component_weight)
    total = sum(component_weight)

    prompt = (
        f"Here is an instruction for a multimodal LLM: ' {instruction}"
        f' You need to grade if the response from the model follows each component of the instruction. '
        f'{components_text}'
        f" The response is:' {response}'"
        f' You need to score the response and be strict. The total score ranges from 0 to {total}, depending on if the response follows the instruction. '
        f'{score_text}'
        f' List scores of each component, and the total score in one sentence in this EXACT format: {score_format}.'
        f' Use only numbers for score values. Do not use markdown formatting or asterisks. Then explain your reasons.'
    )
    return prompt


def parse_mia_score(component_type: List[str], raw_score: str) -> Dict[str, float]:
    """
    Parse the LLM judge raw score string into a score dict.

    Equivalent to reference.py process_rawscore() but standalone.

    Returns:
        dict mapping component_type entries to normalized [0,1] scores,
        plus 'total_score' key.
    """
    score_dict: Dict[str, float] = {}

    try:
        if not component_type or not isinstance(component_type, list):
            logger.error(f'Invalid component_type: {component_type}')
            return {'total_score': 0.0}

        if not raw_score or not isinstance(raw_score, str):
            logger.error(f'Invalid raw_score: {raw_score}')
            for comp in component_type:
                score_dict[comp] = 0.0
            score_dict['total_score'] = 0.0
            return score_dict

        # Pattern: "score of component X: Y/Z" or "component X: Y/Z"
        component_pattern = r'(?:score\s+of\s+)?component\s+(\d+)\s*:\s*(\d+)\s*/\s*(\d+)'
        total_pattern = r'total\s+score\s*:\s*(\d+)\s*/\s*(\d+)'

        # Extract per-component scores
        try:
            component_matches = re.findall(component_pattern, raw_score, re.IGNORECASE)
            for match in component_matches:
                try:
                    component_num = int(match[0]) - 1  # 0-based index
                    if 0 <= component_num < len(component_type):
                        numerator = int(match[1].strip())
                        denominator = int(match[2].strip())
                        score = numerator / denominator if denominator != 0 else 0.0
                        score = max(0.0, min(1.0, score))
                        score_dict[component_type[component_num]] = score
                    else:
                        logger.warning(
                            f'Component number {component_num + 1} out of range for {len(component_type)} components'
                        )
                except (ValueError, IndexError) as e:
                    logger.warning(f'Error parsing component match {match}: {e}')
                    continue
        except Exception as e:
            logger.error(f'Error in component pattern matching: {e}')

        # Extract total score
        try:
            total_match = re.search(total_pattern, raw_score, re.IGNORECASE)
            if total_match:
                total_numerator = int(total_match.group(1).strip())
                total_denominator = int(total_match.group(2).strip())
                total_score = total_numerator / total_denominator if total_denominator != 0 else 0.0
                total_score = max(0.0, min(1.0, total_score))
                score_dict['total_score'] = total_score
            else:
                # Fallback: average of component scores
                if score_dict:
                    score_dict['total_score'] = sum(score_dict.values()) / len(score_dict)
                else:
                    score_dict['total_score'] = 0.0
        except Exception as e:
            logger.error(f'Error parsing total score: {e}')
            score_dict['total_score'] = 0.0

        # Ensure all components have a score entry
        for comp in component_type:
            if comp not in score_dict:
                logger.warning(f'Missing score for component: {comp}')
                score_dict[comp] = 0.0

        if 'total_score' not in score_dict:
            component_values = [v for v in score_dict.values()]
            score_dict['total_score'] = sum(component_values) / len(component_values) if component_values else 0.0

    except Exception as e:
        logger.error(f'Unexpected error in parse_mia_score: {e}')
        score_dict = {comp: 0.0 for comp in (component_type or [])}
        score_dict['total_score'] = 0.0

    # Final validation: clamp all values to [0, 1]
    for key in list(score_dict.keys()):
        val = score_dict[key]
        if not isinstance(val, (int, float)) or val < 0 or val > 1:
            logger.warning(f'Invalid score for {key}: {val}, setting to 0')
            score_dict[key] = 0.0

    return score_dict
