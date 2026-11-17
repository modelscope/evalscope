import json
import unittest
from typing import Dict

from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.deepsearchqa.utils import (
    GRADE_CONTRACT,
    _answer_keys,
    _reference_parts,
    aggregate_official_scores,
    metrics_from_grade,
    rule_fallback_score,
)
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus
from tests.api.judge.helpers import ScriptedJudge, make_state


class TestDeepSearchQAUtils(unittest.TestCase):

    def test_rule_fallback_handles_single_answer_substring(self):
        value, metadata = rule_fallback_score('The answer is Aotearoa.', 'Aotearoa', 'Single Answer')

        self.assertEqual(metadata['correct'], 1)
        self.assertEqual(metadata['expected'], 1)
        self.assertEqual(value['f1'], 1.0)

    def test_rule_fallback_does_not_match_empty_reference_part(self):
        value, metadata = rule_fallback_score('', '', 'Single Answer')

        self.assertEqual(metadata['correct'], 0)
        self.assertEqual(metadata['expected'], 0)
        self.assertEqual(value['f1'], 0.0)

    def test_rule_fallback_accepts_reordered_set_answers(self):
        value, metadata = rule_fallback_score('France; Belgium', 'Belgium, France', 'Set Answer')

        self.assertEqual(metadata['correct'], 2)
        self.assertEqual(metadata['excessive'], 0)
        self.assertEqual(value['f1'], 1.0)

    def test_rule_fallback_does_not_count_missing_set_answers_as_excessive(self):
        value, metadata = rule_fallback_score('Belgium', 'Belgium, France', 'Set Answer')

        self.assertEqual(metadata['correct'], 1)
        self.assertEqual(metadata['expected'], 2)
        self.assertEqual(metadata['excessive'], 0)
        self.assertEqual(value['precision'], 1.0)
        self.assertEqual(value['recall'], 0.5)

    def test_grade_contract_rejects_fences_but_accepts_the_same_json_object(self) -> None:
        judge_response = """
        ```json
        {
          "Answer Correctness": {
            "Explanation": "Both answers are present.",
            "Correctness Details": {"Belgium": true, "France": true},
            "Excessive Answers": []
          }
        }
        ```
        """

        result = GRADE_CONTRACT.parse(judge_response)
        self.assertFalse(result.ok)
        result = GRADE_CONTRACT.parse(judge_response.strip().removeprefix('```json').removesuffix('```').strip())
        self.assertTrue(result.ok)
        value, metadata = metrics_from_grade(result.value)

        self.assertEqual(value['f1'], 1.0)
        self.assertEqual(metadata['correctness_details'], {'Belgium': True, 'France': True})

    def test_grade_contract_rejects_json_embedded_in_prose(self) -> None:
        """Explanations belong in the declared JSON field, not outside the object."""
        payload = {
            'Answer Correctness': {
                'Explanation': 'Only one expected answer is present.',
                'Correctness Details': {
                    'Belgium': True,
                    'France': False
                },
                'Excessive Answers': ['Italy'],
            }
        }

        result = GRADE_CONTRACT.parse(f'Rating follows:\n{json.dumps(payload)}')
        self.assertFalse(result.ok)
        result = GRADE_CONTRACT.parse(json.dumps(payload))
        self.assertTrue(result.ok)
        value, _ = metrics_from_grade(result.value)

        self.assertEqual(value['precision'], 0.5)
        self.assertEqual(value['recall'], 0.5)

    def test_grade_contract_rejects_unknown_boolean_strings(self):
        payload = {
            'Answer Correctness': {
                'Explanation': 'Malformed flag.',
                'Correctness Details': {
                    'Belgium': 'maybe'
                },
                'Excessive Answers': [],
            }
        }

        result = GRADE_CONTRACT.parse(json.dumps(payload))

        self.assertFalse(result.ok)

    def _judge_score(
        self, details: Dict[str, bool], reference: str = 'Belgium, France', answer_type: str = 'Set Answer'
    ) -> Score:
        adapter = get_benchmark('deepsearchqa', TaskConfig(
            model='m', datasets=['deepsearchqa'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]},
        ))
        payload = {
            'Answer Correctness': {
                'Explanation': 'Controlled verdict.',
                'Correctness Details': details,
                'Excessive Answers': [],
            }
        }
        adapter.llm_judge = ScriptedJudge([json.dumps(payload)])
        state = make_state('Belgium', reference)
        state.metadata = {'answer_type': answer_type}
        return adapter.score_with_judge_contracts('Belgium', 'Belgium', reference, state)

    def test_complete_partial_verdict_keeps_the_dataset_denominator(self) -> None:
        score = self._judge_score({'Belgium': True, 'France': False})

        self.assertTrue(score.status.is_usable)
        self.assertEqual(score.value['precision'], 1.0)
        self.assertEqual(score.value['recall'], 0.5)
        self.assertEqual(score.value['f1'], 2 / 3)

    def test_missing_or_unknown_expected_answer_is_excluded(self) -> None:
        for details in ({'Belgium': True}, {'Belgium': True, 'France': True, 'Italy': False}):
            with self.subTest(details=details):
                score = self._judge_score(details)

                self.assertEqual(score.status, ScoreStatus.EXCLUDED)
                self.assertEqual(score.value, {})
                self.assertEqual(score.judge_summary.failures, {'parse_error': 1})

    def test_repeated_answers_keep_grouped_occurrences_in_the_dataset_denominator(self) -> None:
        reference = (
            '2021: Brazil, United States, China, Indonesia, India\n'
            '2020: Brazil, United States, China, Indonesia, India\n'
            '2019: Brazil, United States, China, Indonesia, India'
        )
        parts = _reference_parts(reference, 'Set Answer')
        keys = _answer_keys(parts)

        self.assertEqual(len(parts), 15)
        self.assertEqual(len(set(keys)), 15)
        score = self._judge_score({key: True for key in keys}, reference)
        self.assertTrue(score.status.is_usable)
        self.assertEqual(score.value['f1'], 1.0)

        incomplete = {key: True for key in keys[:-1]}
        score = self._judge_score(incomplete, reference)
        self.assertEqual(score.status, ScoreStatus.EXCLUDED)
        self.assertEqual(score.value, {})

    def test_single_answer_comma_is_not_split_into_dataset_keys(self) -> None:
        score = self._judge_score({'Washington': True}, 'Washington, D.C.', 'Single Answer')

        self.assertEqual(score.status, ScoreStatus.EXCLUDED)
        self.assertEqual(score.value, {})

    def test_reference_split_preserves_parentheses_and_thousands_separators(self) -> None:
        value, metadata = rule_fallback_score(
            '', 'First responders (fire, police), More than 1,500', 'Set Answer'
        )

        self.assertEqual(metadata['expected'], 2)
        self.assertEqual(value['recall'], 0.0)

    def test_reference_split_ignores_a_list_conjunction(self) -> None:
        _, metadata = rule_fallback_score('C', 'A, B, and C', 'Set Answer')

        self.assertEqual(metadata['correct'], 1)
        self.assertEqual(metadata['expected'], 3)

    def test_reference_split_handles_top_level_numeric_and_parenthesized_lists(self) -> None:
        cases = [
            ('2023', '2017 and 2023'),
            ('San Antonio (Texas)', 'Phoenix (Arizona) and San Antonio (Texas)'),
        ]
        for prediction, reference in cases:
            with self.subTest(reference=reference):
                _, metadata = rule_fallback_score(prediction, reference, 'Set Answer')

                self.assertEqual(metadata['correct'], 1)
                self.assertEqual(metadata['expected'], 2)

    def test_reference_split_preserves_numeric_comma_and_name_conjunctions(self) -> None:
        cases = [
            'Final score: 4,0 Team Vitality',
            'Bosnia and Herzegovina',
            'Virginia Polytechnic Institute and State University',
        ]
        for reference in cases:
            with self.subTest(reference=reference):
                value, metadata = rule_fallback_score(reference, reference, 'Set Answer')

                self.assertEqual(metadata['correct'], 1)
                self.assertEqual(metadata['expected'], 1)
                self.assertEqual(value['f1'], 1.0)

    def test_aggregate_scores_excludes_empty_and_failed_responses_from_means(self):
        sample_scores = [
            SampleScore(
                sample_id=0,
                score=Score(value={
                    'precision': 1.0,
                    'recall': 0.5,
                    'f1': 2 / 3
                }, metadata={}),
            ),
            SampleScore(sample_id=1, score=Score(value={}, metadata={'empty_model_response': True})),
            SampleScore(sample_id=2, score=Score(value={}, status=ScoreStatus.EXCLUDED)),
        ]

        scores = {
            f'{score.aggregation}_{score.metric_name}': score
            for score in aggregate_official_scores(sample_scores)
        }

        self.assertEqual(scores['mean_precision'].num, 1)
        self.assertEqual(scores['rate_empty_model_response'].score, 1 / 3)
        self.assertEqual(scores['rate_judge_parse_failure'].score, 1 / 3)


if __name__ == '__main__':
    unittest.main()
