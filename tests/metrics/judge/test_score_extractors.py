"""Unit tests for LLM-judge score extraction strategies.

These lock in the documented fail-close contract: a failed judge request
(``[ERROR] ...`` responses) and any unmatched/unmapped extraction must score
0, never full credit. Covers ``PatternScoreExtractor``, ``NumericScoreExtractor``
and the ``[ERROR]`` guard in ``LLMJudge.get_score``.
"""

from evalscope.metrics.judge.llm_judge import JUDGE_ERROR_PREFIX, LLMJudge
from evalscope.metrics.judge.score_extractors import NumericScoreExtractor, PatternScoreExtractor


class TestPatternScoreExtractor:
    def test_mapped_answer_returns_mapping_value(self) -> None:
        extractor = PatternScoreExtractor(pattern=r'\b([AB])\b', score_mapping={'A': 1.0, 'B': 0.5})
        assert extractor.extract('The answer is A.') == 1.0
        assert extractor.extract('The answer is B.') == 0.5

    def test_no_match_returns_zero(self) -> None:
        extractor = PatternScoreExtractor(pattern=r'\b([AB])\b')
        assert extractor.extract('The answer is C.') == 0.0

    def test_matched_but_unmapped_returns_zero(self) -> None:
        # 'B' matches the pattern but is not in the mapping: fail closed at 0.0
        extractor = PatternScoreExtractor(pattern=r'\b([AB])\b', score_mapping={'A': 1.0})
        assert extractor.extract('The answer is B.') == 0.0

    def test_error_response_returns_zero_for_letter_pattern(self) -> None:
        # An [ERROR] response carries no letter token; extraction fails closed.
        extractor = PatternScoreExtractor(pattern=r'\b([AB])\b')
        assert extractor.extract(f'{JUDGE_ERROR_PREFIX} Qwen3-235B@http://127.0.0.1:8000/v1/ HTTP 500') == 0.0


class TestNumericScoreExtractor:
    def test_in_range_value_passes_through(self) -> None:
        extractor = NumericScoreExtractor(pattern=r'\[\[(.*?)\]\]')
        assert extractor.extract('[[0.5]]') == 0.5

    def test_out_of_range_high_clamps_to_max(self) -> None:
        # A 0-100-scale judge reporting [[90]] must not become full credit silently.
        extractor = NumericScoreExtractor(pattern=r'\[\[(.*?)\]\]')
        assert extractor.extract('[[90]]') == 1.0

    def test_out_of_range_low_clamps_to_min(self) -> None:
        extractor = NumericScoreExtractor(pattern=r'\[\[(.*?)\]\]')
        assert extractor.extract('[[-1]]') == 0.0

    def test_no_match_returns_zero(self) -> None:
        extractor = NumericScoreExtractor(pattern=r'\[\[(.*?)\]\]')
        assert extractor.extract('no brackets here') == 0.0


class TestLLMJudgeErrorGuard:
    def make_judge(self, score_type: str, **kwargs) -> LLMJudge:
        return LLMJudge(score_type=score_type, **kwargs)

    def test_none_response_returns_zero(self) -> None:
        judge = self.make_judge('pattern', score_pattern=r'\b([AB])\b')
        assert judge.get_score(None) == 0.0

    def test_error_response_returns_zero_even_with_loose_numeric_pattern(self) -> None:
        # Without the [ERROR] guard, a loose numeric pattern would parse the
        # embedded digits (500, 8000, 127) from a failed judge response and clamp
        # them to full credit — the opposite of the fail-close contract.
        judge = self.make_judge('numeric', score_pattern=r'(\d+)')
        response = f'{JUDGE_ERROR_PREFIX} Qwen3-235B@http://127.0.0.1:8000/v1/ HTTP 500'
        assert judge.get_score(response) == 0.0

    def test_normal_response_still_extracts(self) -> None:
        judge = self.make_judge('numeric', score_pattern=r'\[\[(.*?)\]\]')
        assert judge.get_score('[[0.75]]') == 0.75

    def test_judge_returns_error_prefix_when_generate_raises(self) -> None:
        # Locks the producer contract: on a failed request, ``judge`` returns a
        # ``JUDGE_ERROR_PREFIX``-prefixed string that ``get_score`` then fails closed on.
        judge = self.make_judge('pattern', score_pattern=r'\b([AB])\b')

        class _FailingModel:
            def generate(self, messages):
                raise RuntimeError('connection refused')

        judge.model = _FailingModel()
        result = judge.judge(prompt='question')
        assert result.startswith(JUDGE_ERROR_PREFIX)
