"""
EQ-Bench Answer Validation and Scoring Logic

This module implements the official EQ-Bench scoring algorithm for evaluating
language models on emotional intelligence tasks. It is based on the reference
implementation from the official EQ-Bench repository.

This module is responsible for:
1. Parsing emotion intensity scores from model outputs
2. Validating answer format correctness
3. Calculating difference scores against reference answers
4. Supporting both v1 (normalized) and v2 (full-scale) scoring systems

Source: Based on the official EQ-Bench reference implementation
Reference: https://github.com/eqbench/eq-bench
Paper: https://arxiv.org/abs/2312.06281

Note: This file is bundled with the evalscope package to ensure it's available
in both development (pip install -e .) and production (pip install .) installations.
"""

import math
import re

from evalscope.utils.logger import get_logger

logger = get_logger()


def parse_answers(text, REVISE):
    """
    Parse emotion intensity scores from model inference output (English format).

    Expected format:
    - Without revision: emotion1: score1\nemotionN: scoreN
    - With revision: First pass scores: ... Revised scores: ...

    Args:
        text: Raw output text from the model
        REVISE: Whether revision mode is enabled

    Returns:
        tuple: (first_pass_answers, revised_answers) Two dictionaries containing
               emotion names as keys and scores as values
    """
    first_pass_answers = {}
    revised_answers = {}

    # Remove markdown formatting
    text = text.replace('*', '').replace('#', '')

    # Extract first pass scores
    if REVISE:
        first_pass_match = re.search(r'First pass scores:(.*?)Revised scores:', text, re.DOTALL)
        if first_pass_match:
            first_pass_text = first_pass_match.group(1)
            first_pass_answers = dict(re.findall(r'(\w+):\s+(\d+)', first_pass_text))

        # Extract revised scores
        revised_match = re.search(r'Revised scores:(.*?)$', text, re.DOTALL)
        if revised_match:
            revised_text = revised_match.group(1)
            revised_answers = dict(re.findall(r'(\w+):\s+(\d+)', revised_text))
    else:
        first_pass_answers = dict(re.findall(r'(\w+):\s+(\d+)', text))
        revised_answers = {}

    return first_pass_answers, revised_answers


def parse_answers_de(text, REVISE):
    """
    Parse emotion intensity scores from model inference output (German format).

    Args:
        text: Raw output text from the model
        REVISE: Whether revision mode is enabled

    Returns:
        tuple: (first_pass_answers, revised_answers) Two dictionaries containing
               emotion names as keys and scores as values
    """
    first_pass_answers = {}
    revised_answers = {}

    # Remove markdown formatting
    text = text.replace('*', '').replace('#', '')

    first_pass_heading_pattern = r'(Erste.*?):\s*(.*?)(?=Überarbeitete|$)'
    revised_heading_pattern = r'(Überarbeitete.*?):\s*(.*)'

    if REVISE:
        first_pass_match = re.search(first_pass_heading_pattern, text, re.IGNORECASE | re.DOTALL)
        if first_pass_match:
            first_pass_text = first_pass_match.group(2)
            pairs = re.findall(r'([a-zA-ZäöüßÄÖÜ\s]+):\s*\**(\d+(?:,\d+)?)\**', first_pass_text)
            first_pass_answers = {label.strip(): score.replace('*', '') for label, score in pairs}

        revised_match = re.search(revised_heading_pattern, text, re.IGNORECASE | re.DOTALL)
        if revised_match:
            revised_text = revised_match.group(2)
            pairs = re.findall(r'([a-zA-ZäöüßÄÖÜ\s]+):\s*\**(\d+(?:,\d+)?)\**', revised_text)
            revised_answers = {label.strip(): score.replace('*', '') for label, score in pairs}
    else:
        pairs = re.findall(r'([a-zA-ZäöüßÄÖÜ\s]+):\s*\**(\d+(?:,\d+)?)\**', text)
        first_pass_answers = {label.strip(): score.replace('*', '') for label, score in pairs}
        revised_answers = {}

    return first_pass_answers, revised_answers


def calculate_score_fullscale(reference, user):
    """
    Calculate score for a single question using the v2 full-scale scoring system.

    Scoring rules:
    1. Validate that the user provided exactly 4 emotion scores and that
       the emotion types match the reference answer
    2. Calculate the difference between predicted and reference values for each emotion
    3. For differences <= 5, apply an S-shaped scaling function
    4. For differences > 5, use linear scaling
    5. Apply an adjustment constant so that random answers score 0

    Args:
        reference: Reference answer dictionary containing emotion1-4 and
                   their corresponding _score fields
        user: User answer dictionary with format {emotion: score}

    Returns:
        float: The score for this question (0-10 range), or None if parsing fails
    """
    # First check if emotions match the reference answer
    if len(user.items()) != 4:
        return None

    emotions_dict = {}
    for emotion, user_emotion_score in user.items():
        for i in range(1, 5):
            if emotion.lower() == reference[f'emotion{i}'].lower():
                emotions_dict[emotion.lower()] = True

    if len(emotions_dict) != 4:
        logger.error('! Error: emotions did not match reference')
        logger.error(f'{user}')
        return None

    difference_tally = 0  # Accumulated difference from reference answer

    # Iterate through each emotion in the user's answer
    for emotion, user_emotion_score in user.items():
        # If this emotion is in the reference answer, calculate the difference
        for i in range(1, 5):
            if emotion.lower() == reference[f'emotion{i}'].lower():
                d = abs(float(user_emotion_score) - float(reference[f'emotion{i}_score']))
                # d ranges from 0 to 10
                if d == 0:
                    scaled_difference = 0
                elif d <= 5:
                    # S-shaped scaling function
                    # Formula visualization: https://www.desmos.com/calculator
                    # 6.5 * 1 / (1 + e^(-1.2 * (x - 4)))
                    scaled_difference = 6.5 * (1 / (1 + math.e**(-1.2 * (d - 4))))
                else:
                    scaled_difference = d
                difference_tally += scaled_difference

    # Invert the accumulated difference so that answers closer to reference score higher
    # The adjustment constant is chosen so that random answers score 0
    adjust_const = 0.7477
    final_score = 10 - (difference_tally * adjust_const)

    return final_score


def calculate_score(reference, user):
    """
    Calculate score for a single question using the v1 normalized scoring system (legacy version).

    Scoring rules:
    1. Validate that the user provided exactly 4 emotion scores and that
       the emotion types match the reference answer
    2. Normalize the user's scores so that they sum to 10
    3. Calculate the absolute difference between normalized scores and reference scores
    4. Use a fixed adjustment constant so that random answers score 0

    Args:
        reference: Reference answer dictionary
        user: User answer dictionary

    Returns:
        float: The score for this question, or None if parsing fails
    """
    # First check if emotions match the reference answer
    if len(user.items()) != 4:
        logger.error('! Error: 4 emotions were not returned')
        logger.error(f'{user}')
        return None

    emotions_dict = {}
    for emotion, user_emotion_score in user.items():
        for i in range(1, 5):
            if emotion.lower() == reference[f'emotion{i}'].lower():
                emotions_dict[emotion] = True

    if len(emotions_dict) != 4:
        logger.error('! Error: emotions did not match reference')
        logger.error(f'{user}')
        return None

    # Normalize the user's scores so that they sum to 10
    total_user_score = sum(float(score) for score in user.values())
    if total_user_score <= 0:
        logger.error('Error: total of scores must be > 0')
        logger.error(f'{user}')
        return None
    user = {emotion: float(score) / total_user_score * 10 for emotion, score in user.items()}

    difference_tally = 0  # Accumulated difference from reference answer

    # Iterate through each emotion in the user's answer
    for emotion, user_emotion_score in user.items():
        # If this emotion is in the reference answer, calculate the difference
        for i in range(1, 5):
            if emotion == reference[f'emotion{i}']:
                difference_tally += abs(user_emotion_score - reference[f'emotion{i}_score'])

    # Invert the accumulated difference so that answers closer to reference score higher
    # Subtract from 10 because this constant makes random answers score 0, which is a useful baseline
    final_score = 10 - difference_tally

    return final_score


def validate_answer_format(user_answers, reference_emotions):
    """
    Validate that the user's answer format is correct.

    Args:
        user_answers: User answer dictionary with format {emotion: score}
        reference_emotions: List of emotions from the reference answer

    Returns:
        tuple: (is_valid, error_message) where is_valid is a boolean indicating
               whether the format is valid, and error_message contains details
               about any validation errors found
    """
    # Check if there are exactly 4 emotions
    if len(user_answers) != 4:
        return False, f'Expected 4 emotions, got {len(user_answers)}'

    # Check if emotions match the reference
    user_emotions = set(e.lower() for e in user_answers.keys())
    ref_emotions = set(e.lower() for e in reference_emotions)

    if user_emotions != ref_emotions:
        missing = ref_emotions - user_emotions
        extra = user_emotions - ref_emotions
        error_msg = ''
        if missing:
            error_msg += f'Missing emotions: {missing}. '
        if extra:
            error_msg += f'Extra emotions: {extra}.'
        return False, error_msg

    # Check if scores are valid numbers
    try:
        for emotion, score in user_answers.items():
            score_float = float(score)
            if score_float < 0 or score_float > 10:
                return False, f'Score for {emotion} out of range (0-10): {score_float}'
    except ValueError:
        return False, f'Invalid score format: {user_answers}'

    return True, ''


# Scoring system documentation
SCORING_SYSTEM_INFO = """
EQ-Bench Scoring System Documentation:

v2 Full-Scale Scoring System (Recommended):
- Directly compares predicted emotion intensities (0-10) with reference answers
- Uses S-shaped scaling function for small differences (≤5), linear scaling for large differences (>5)
- Adjustment constant 0.7477 ensures random answers score 0
- Final score range: 0-100 (100 = perfect match with reference answer)

v1 Normalized Scoring System (Legacy):
- Normalizes predicted scores so they sum to 10, only compares relative intensities
- Reduces sensitivity to absolute intensity prediction biases
- However, cannot evaluate the model's ability to judge absolute emotion intensities
- v1 scores and v2 scores are not directly comparable
"""
