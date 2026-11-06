"""Mixed-script MER must retain Chinese character boundaries after English words."""

import pytest

from evalscope.metrics.audio.metrics import MER


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('hello世界', ['hello', '世', '界']),
        ('世界hello', ['世', '界', 'hello']),
        ('中文hello世界AI模型', ['中', '文', 'hello', '世', '界', 'AI', '模', '型']),
        ("don't中文a1_b2世界", ["don't", '中', '文', 'a1_b2', '世', '界']),
        ('A\u4e00B\u9fffC', ['A', '\u4e00', 'B', '\u9fff', 'C']),
        ("don't O'Reilly rock'n'roll a1_b2", ["don't", "O'Reilly", "rock'n'roll", 'a1_b2']),
        ("rock2'n", ['rock2', "'n"]),
        ('hello,world!\n世界\tAI', ['hello', 'world', '世', '界', 'AI']),
        ('123 _ 你好', ['你', '好']),
        ('café Aβ2 A٣', ['café', 'Aβ2', 'A٣']),
        ('A㐀B 㐀AB', ['A㐀B', 'AB']),
        (' \t\n', []),
    ],
)
def test_mer_tokenization(text: str, expected: list[str]) -> None:
    assert MER._tokenize(text) == expected


@pytest.mark.parametrize(
    ('prediction', 'reference', 'expected'),
    [
        ('hello世界', 'hello 世界', 0.0),
        ('hello 世界', 'hello世界', 0.0),
        ('hello，世界', 'hello世界', 0.0),
        ('hello世介', 'hello世界', 1 / 3),
        ('hello世', 'hello世界', 1 / 3),
        ('hello世大界', 'hello世界', 1 / 4),
        ('世界', '世介', 1 / 2),
        ('hello there', 'hello world', 1 / 2),
        ('Hello', 'hello', 1.0),
        ('', '', 0.0),
    ],
)
def test_mer_scores_token_errors(prediction: str, reference: str, expected: float) -> None:
    assert MER()(prediction, reference) == pytest.approx(expected)
