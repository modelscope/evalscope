import json
import unittest

from evalscope.api.metric import SampleScore, Score
from evalscope.benchmarks.deepsearchqa.utils import (
    GRADE_CONTRACT,
    aggregate_official_scores,
    metrics_from_grade,
    rule_fallback_score,
)
from evalscope.constants import ScoreStatus


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

    def test_grade_contract_parses_the_official_json_fence(self):
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
        self.assertTrue(result.ok)
        value, metadata = metrics_from_grade(result.value)

        self.assertEqual(value['f1'], 1.0)
        self.assertEqual(metadata['correctness_details'], {'Belgium': True, 'France': True})

    def test_grade_contract_accepts_json_embedded_in_prose(self):
        """A reasoning judge may explain before the JSON; the contract reads the single object."""
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
