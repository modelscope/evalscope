# Copyright (c) Alibaba, Inc. and its affiliates.

import re


def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)

    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)

    total_count = chinese_char_count + english_word_count

    is_chinese = chinese_char_count > english_word_count

    return total_count, is_chinese


def chinese_to_arabic(chinese_number: str) -> int:
    chinese_numerals = {
        '零': 0,
        '一': 1,
        '二': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9,
        '俩': 2,
        '两': 2,
    }

    return chinese_numerals.get(chinese_number, chinese_number)
