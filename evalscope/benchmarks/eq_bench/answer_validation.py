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

import re
import math


def parse_answers(text, REVISE):
    """
    解析英文的情感强度评分（从模型推理输出中提取）

    期望格式：
    - 不带修订：emotion1: score1\nemotionN: scoreN
    - 带修订：First pass scores: ... Revised scores: ...

    Args:
        text: 模型的原始输出文本
        REVISE: 是否启用修订模式

    Returns:
        tuple: (first_pass_answers, revised_answers) 两个字典
    """
    first_pass_answers = {}
    revised_answers = {}

    # 去除 markdown 格式
    text = text.replace('*', '').replace('#', '')

    # 提取首次评分
    if REVISE:
        first_pass_match = re.search(r'First pass scores:(.*?)Revised scores:', text, re.DOTALL)
        if first_pass_match:
            first_pass_text = first_pass_match.group(1)
            first_pass_answers = dict(re.findall(r'(\w+):\s+(\d+)', first_pass_text))

        # 提取修订后的评分
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
    解析德语的情感强度评分

    Args:
        text: 模型的原始输出文本
        REVISE: 是否启用修订模式

    Returns:
        tuple: (first_pass_answers, revised_answers) 两个字典
    """
    first_pass_answers = {}
    revised_answers = {}

    # 去除 markdown 格式
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
    使用 v2 全尺度评分系统计算单个问题的得分

    评分规则：
    1. 验证用户提供了4个情感评分，且情感类型与参考答案匹配
    2. 计算每个情感的预测值与参考值之间的差异
    3. 对于差异 <= 5 的，使用 S 形缩放函数
    4. 对于差异 > 5 的，使用线性缩放
    5. 调整常数设置为使随机回答得分为 0

    Args:
        reference: 参考答案字典，包含 emotion1-4 和对应的 _score
        user: 用户答案字典，格式为 {emotion: score}

    Returns:
        float: 该问题的得分 (0-10 范围)，或 None（如果解析失败）
    """
    # 首先检查情感是否与参考答案匹配
    if len(user.items()) != 4:
        return None

    emotions_dict = {}
    for emotion, user_emotion_score in user.items():
        for i in range(1, 5):
            if emotion.lower() == reference[f'emotion{i}'].lower():
                emotions_dict[emotion.lower()] = True

    if len(emotions_dict) != 4:
        print('! Error: emotions did not match reference')
        print(user)
        return None

    difference_tally = 0  # 与参考答案的差异累计

    # 遍历用户答案中的每个情感
    for emotion, user_emotion_score in user.items():
        # 如果该情感在参考答案中，计算差异
        for i in range(1, 5):
            if emotion.lower() == reference[f'emotion{i}'].lower():
                d = abs(float(user_emotion_score) - float(reference[f'emotion{i}_score']))
                # d 的范围是 0 到 10
                if d == 0:
                    scaled_difference = 0
                elif d <= 5:
                    # S 形缩放函数
                    # https://www.desmos.com/calculator
                    # 6.5\cdot\ \frac{1}{\left(1\ +\ e^{\left(-1.2\cdot\left(x-4\right)\right)}\right)}
                    scaled_difference = 6.5 * (1 / (1 + math.e ** (-1.2 * (d-4))))
                else:
                    scaled_difference = d
                difference_tally += scaled_difference

    # 反转差异累计，使答案越接近参考答案得分越高
    # 调整常数的选择使得随机回答得分为 0
    adjust_const = 0.7477
    final_score = 10 - (difference_tally * adjust_const)

    return final_score


def calculate_score(reference, user):
    """
    使用 v1 归一化评分系统计算单个问题的得分（遗留版本）

    评分规则：
    1. 验证用户提供了4个情感评分，且情感类型与参考答案匹配
    2. 将用户的评分归一化，使总和为 10
    3. 计算归一化后的评分与参考答案的绝对差异
    4. 使用固定常数调整，使随机回答得分为 0

    Args:
        reference: 参考答案字典
        user: 用户答案字典

    Returns:
        float: 该问题的得分，或 None（如果解析失败）
    """
    # 首先检查情感是否与参考答案匹配
    if len(user.items()) != 4:
        print('! Error: 4 emotions were not returned')
        print(user)
        return None

    emotions_dict = {}
    for emotion, user_emotion_score in user.items():
        for i in range(1, 5):
            if emotion.lower() == reference[f'emotion{i}'].lower():
                emotions_dict[emotion] = True

    if len(emotions_dict) != 4:
        print('! Error: emotions did not match reference')
        print(user)
        return None

    # 将用户的评分归一化，使总和为 10
    total_user_score = sum(float(score) for score in user.values())
    if total_user_score <= 0:
        print('Error: total of scores must be > 0')
        print(user)
        return None
    user = {emotion: float(score) / total_user_score * 10 for emotion, score in user.items()}

    difference_tally = 0  # 与参考答案的差异累计

    # 遍历用户答案中的每个情感
    for emotion, user_emotion_score in user.items():
        # 如果该情感在参考答案中，计算差异
        for i in range(1, 5):
            if emotion == reference[f'emotion{i}']:
                difference_tally += abs(user_emotion_score - reference[f'emotion{i}_score'])

    # 反转差异累计，使答案越接近参考答案得分越高
    # 从 10 减去是因为这个常数使得随机回答得分为 0，这是一个有用的基准
    final_score = 10 - difference_tally

    return final_score


def validate_answer_format(user_answers, reference_emotions):
    """
    验证用户答案的格式是否正确

    Args:
        user_answers: 用户答案字典 {emotion: score}
        reference_emotions: 参考答案中的情感列表

    Returns:
        tuple: (is_valid, error_message)
    """
    # 检查是否有4个情感
    if len(user_answers) != 4:
        return False, f"Expected 4 emotions, got {len(user_answers)}"

    # 检查情感是否匹配
    user_emotions = set(e.lower() for e in user_answers.keys())
    ref_emotions = set(e.lower() for e in reference_emotions)

    if user_emotions != ref_emotions:
        missing = ref_emotions - user_emotions
        extra = user_emotions - ref_emotions
        error_msg = ""
        if missing:
            error_msg += f"Missing emotions: {missing}. "
        if extra:
            error_msg += f"Extra emotions: {extra}."
        return False, error_msg

    # 检查分数是否为有效数字
    try:
        for emotion, score in user_answers.items():
            score_float = float(score)
            if score_float < 0 or score_float > 10:
                return False, f"Score for {emotion} out of range (0-10): {score_float}"
    except ValueError:
        return False, f"Invalid score format: {user_answers}"

    return True, ""


# 评分系统说明
SCORING_SYSTEM_INFO = """
EQ-Bench 评分系统说明：

v2 全尺度评分系统（推荐）：
- 直接比较预测的情感强度（0-10）与参考答案
- 使用 S 形缩放函数处理小差异（≤5），线性处理大差异（>5）
- 调整常数 0.7477 使随机回答得分为 0
- 最终得分范围：0-100（100 = 完美匹配参考答案）

v1 归一化评分系统（遗留）：
- 将预测评分归一化使总和为 10，只比较相对强度
- 减少了对绝对强度预测偏差的敏感性
- 但无法评估模型对绝对情感强度的判断能力
- v1 得分与 v2 得分不可直接比较
"""
