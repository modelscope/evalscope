import json
import unittest

from evalscope.api.metric import SampleScore, Score
from evalscope.benchmarks.deepsearchqa.utils import aggregate_official_scores, parse_judge_response, rule_fallback_score


class TestDeepSearchQAUtils(unittest.TestCase):
    def test_rule_fallback_handles_single_answer_aliases(self):
        value, metadata = rule_fallback_score('The answer is Aotearoa.', ['New Zealand', 'Aotearoa'], 'Single Answer')

        self.assertEqual(metadata['correct'], 1)
        self.assertEqual(metadata['expected'], 1)
        self.assertEqual(value['f1_score'], 1.0)

    def test_rule_fallback_does_not_match_empty_reference_part(self):
        value, metadata = rule_fallback_score('', [''], 'Single Answer')

        self.assertEqual(metadata['correct'], 0)
        self.assertEqual(metadata['expected'], 0)
        self.assertEqual(value['f1_score'], 0.0)

    def test_rule_fallback_accepts_reordered_set_answers(self):
        value, metadata = rule_fallback_score('France; Belgium', 'Belgium, France', 'Set Answer')

        self.assertEqual(metadata['correct'], 2)
        self.assertEqual(metadata['excessive'], 0)
        self.assertEqual(value['f1_score'], 1.0)

    def test_parse_judge_response_handles_common_json_variants(self):
        judge_response = """
        Here is the rating:
        ```JSON
        {
          "Answer Correctness": {
            "Explanation": "Both answers are present.",
            "Correctness Details": {"Belgium": "true", "France": 1},
            "Excessive Answers": []
          }
        }
        ```
        """

        value, metadata = parse_judge_response(judge_response)

        self.assertEqual(value['f1_score'], 1.0)
        self.assertEqual(metadata['correctness_details'], {'Belgium': True, 'France': True})

    def test_parse_judge_response_handles_surrounding_text(self):
        payload = {
            'Answer Correctness': {
                'Explanation': 'Only one expected answer is present.',
                'Correctness Details': {'Belgium': True, 'France': False},
                'Excessive Answers': ['Italy'],
            }
        }

        value, metadata = parse_judge_response(f'Rating follows:\n{json.dumps(payload)}\nDone.')

        self.assertEqual(metadata['correct'], 1)
        self.assertEqual(metadata['expected'], 2)
        self.assertEqual(metadata['excessive'], 1)
        self.assertAlmostEqual(value['precision'], 0.5)

    def test_parse_judge_response_rejects_unknown_boolean_strings(self):
        payload = {
            'Answer Correctness': {
                'Explanation': 'Malformed flag.',
                'Correctness Details': {'Belgium': 'maybe'},
                'Excessive Answers': [],
            }
        }

        value, metadata = parse_judge_response(json.dumps(payload))

        self.assertEqual(value, {})
        self.assertTrue(metadata['invalid_auto_rater_response'])

    def test_aggregate_scores_excludes_empty_and_invalid_responses_from_means(self):
        sample_scores = [
            SampleScore(
                sample_id=0,
                score=Score(value={'precision': 1.0, 'recall': 0.5, 'f1_score': 2 / 3}, metadata={}),
            ),
            SampleScore(sample_id=1, score=Score(value={}, metadata={'empty_model_response': True})),
            SampleScore(sample_id=2, score=Score(value={}, metadata={'invalid_auto_rater_response': True})),
        ]

        scores = {
            f'{score.aggregation_name}_{score.metric_name}': score for score in aggregate_official_scores(sample_scores)
        }

        self.assertEqual(scores['mean_precision'].num, 1)
        self.assertEqual(scores['rate_empty_model_response'].score, 1 / 3)
        self.assertEqual(scores['rate_invalid_auto_rater_response'].score, 1 / 3)


if __name__ == '__main__':
    unittest.main()
