# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the PLawBench rubric grading logic (no network access)."""

import json
import unittest
from unittest.mock import Mock

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.plawbench.utils import parse_json_to_dict
from evalscope.config import TaskConfig
from evalscope.constants import JudgeStrategy

CASE_ANALYSIS_RUBRICS = [
    {
        'criterion': '【结论得分】\n(+5分) 结论正确。',
        'points': '5',
        'tags': '结论得分'
    },
    {
        'criterion': '【案情简述得分】\n(+20分) 关键事实齐备。',
        'points': '20',
        'tags': '案情简述得分'
    },
    {
        'criterion': '【分析过程得分】\n(+20分) 推理完整。',
        'points': '20',
        'tags': '分析过程得分'
    },
    {
        'criterion': '【依据法条得分】\n(+15分) 法条准确。',
        'points': '15',
        'tags': '法条依据得分'
    },
]


def _build_adapter():
    config = TaskConfig(
        model='mock',
        datasets=['plawbench'],
        judge_strategy=JudgeStrategy.LLM,
        judge_model_args={'model_id': 'mock-judge'},
    )
    return get_benchmark('plawbench', config=config)


def _build_task_state(adapter, record):
    sample = adapter.record_to_sample(record)
    return TaskState(model='mock', sample=sample, messages=[], output=None, completed=True)


class TestPLawBenchScoring(unittest.TestCase):

    def setUp(self):
        self.adapter = _build_adapter()
        self.adapter.llm_judge = Mock(model_id='mock-judge')

    def test_parse_json_to_dict_takes_fenced_block(self):
        response = '分析如下：...\n```json\n{"total_points": 8, "max_points": 16,}\n```'
        self.assertEqual(parse_json_to_dict(response), {'total_points': 8, 'max_points': 16})

    def test_parse_json_to_dict_keeps_nested_objects(self):
        response = '```json\n{"score_details": {"结论": {"total_points": 5, "breakdown": [{"a": "}"}]}}}\n```'
        parsed = parse_json_to_dict(response)
        self.assertEqual(parsed['score_details']['结论']['total_points'], 5)

    def test_parse_json_to_dict_prefers_the_trailing_verdict(self):
        response = '示例 {"total_points": 1}\n最终结果：\n```json\n{"total_points": 9, "max_points": 18}\n```'
        self.assertEqual(parse_json_to_dict(response)['total_points'], 9)

    def test_parse_json_to_dict_repairs_unclosed_feedback_array(self):
        """Observed real judge output: the ``suggestions`` array is never closed."""
        response = (
            '```json\n'
            '{\n  "score_details": {"结论": {"total_points": 5, "max_points": 5}},\n'
            '  "overall_feedback": {\n    "suggestions": [\n      "结论应直接回应评分细则。",\n'
            '      "引用法条应全面覆盖核心条款。"\n  }\n}\n```'
        )
        parsed = parse_json_to_dict(response)
        self.assertEqual(parsed['score_details']['结论']['total_points'], 5)

    def test_parse_json_to_dict_repairs_truncated_verdict(self):
        """A verdict cut off at the judge's token limit still yields the emitted sections."""
        response = (
            '```json\n{"score_details": {"结论": {"total_points": 5, "max_points": 5},'
            ' "案件事实": {"total_points": 10, "max_points": 20, "breakdown": [{"rationale": "未完'
        )
        parsed = parse_json_to_dict(response)
        self.assertEqual(parsed['score_details']['案件事实']['total_points'], 10)

    def test_legal_consultation_scores_point_ratio(self):
        record = {
            'id': 'legal_consultation-1',
            'task': 'legal_consultation',
            'judge_type': 'legal_qa',
            'category': '',
            'context': '',
            'question': '律师，我要离婚。',
            'rubrics': '总分：16分\n1.（+3分）是否有报警记录？',
            'max_points': 16,
        }
        task_state = _build_task_state(self.adapter, record)
        self.adapter.llm_judge.judge.return_value = '```json\n{"total_points": 8, "max_points": 16}\n```'

        score = self.adapter.llm_match_score('答案', '答案', '', task_state)

        self.assertAlmostEqual(score.value['acc'], 0.5)
        self.assertEqual(score.main_score_name, 'acc')
        self.assertFalse(score.metadata['judge_failed'])
        # The consultation judge is limited to the first 25 questions at the 'mid' difficulty.
        system_prompt = self.adapter.llm_judge.judge.call_args.kwargs['system_prompt']
        self.assertIn('只对前 25 条问题进行评价', system_prompt)

    def test_awarded_points_are_clamped_to_dataset_max(self):
        record = {
            'id': 'plaintiff_statement-1',
            'task': 'plaintiff_statement',
            'judge_type': 'document_generation',
            'category': '民间借贷纠纷',
            'context': '',
            'question': '请帮我写起诉状。',
            'rubrics': '总分：100\n一、案由（20分）',
            'max_points': 100,
        }
        task_state = _build_task_state(self.adapter, record)
        # The judge over-reports both the award and the denominator.
        self.adapter.llm_judge.judge.return_value = '```json\n{"total_points": 250, "max_points": 500}\n```'

        score = self.adapter.llm_match_score('文书', '文书', '', task_state)

        self.assertAlmostEqual(score.value['acc'], 1.0)
        self.assertAlmostEqual(score.metadata['awarded_points'], 100.0)
        self.assertAlmostEqual(score.metadata['max_points'], 100.0)

    def test_case_analysis_reports_per_dimension_metrics(self):
        record = {
            'id': 'case_analysis-1',
            'task': 'case_analysis',
            'judge_type': 'case_analysis',
            'category': '个人生活',
            'context': '某甲与某公司签订合同。',
            'question': '某甲能否主张精神损害赔偿？',
            'rubrics': json.dumps(CASE_ANALYSIS_RUBRICS, ensure_ascii=False),
            'max_points': 60,
        }
        task_state = _build_task_state(self.adapter, record)
        self.adapter.llm_judge.judge.return_value = json.dumps({
            'score_details': {
                '结论': {
                    'total_points': 5,
                    'max_points': 5
                },
                '案件事实': {
                    'total_points': 10,
                    'max_points': 20
                },
                '推理过程': {
                    'total_points': 0,
                    'max_points': 20
                },
                '法条依据': {
                    'total_points': 15,
                    'max_points': 15
                },
            }
        })

        score = self.adapter.llm_match_score('回答', '回答', '', task_state)

        self.assertAlmostEqual(score.value['conclusion_acc'], 1.0)
        self.assertAlmostEqual(score.value['fact_acc'], 0.5)
        self.assertAlmostEqual(score.value['reasoning_acc'], 0.0)
        self.assertAlmostEqual(score.value['law_acc'], 1.0)
        self.assertAlmostEqual(score.value['acc'], 30 / 60)
        # 'acc' must come first: the dashboard uses the report's first metric as the headline.
        self.assertEqual(next(iter(score.value)), 'acc')
        self.assertIn('30 / 60', task_state.target)

    def test_unparsable_judge_response_is_scored_zero(self):
        record = {
            'id': 'defendant_statement-1',
            'task': 'defendant_statement',
            'judge_type': 'document_generation',
            'category': '买卖合同纠纷',
            'context': '',
            'question': '请帮我写答辩状。',
            'rubrics': '总分：80',
            'max_points': 80,
        }
        task_state = _build_task_state(self.adapter, record)
        self.adapter.llm_judge.judge.return_value = 'Default output from mockllm/model'

        score = self.adapter.llm_match_score('文书', '文书', '', task_state)

        self.assertAlmostEqual(score.value['acc'], 0.0)
        self.assertEqual(next(iter(score.value)), 'acc')
        self.assertTrue(score.metadata['judge_failed'])
        self.assertEqual(self.adapter.llm_judge.judge.call_count, 3)

    def test_rule_strategy_is_rejected(self):
        record = {
            'id': 'case_analysis-1',
            'task': 'case_analysis',
            'judge_type': 'case_analysis',
            'category': '个人生活',
            'context': '案情',
            'question': '问题',
            'rubrics': json.dumps(CASE_ANALYSIS_RUBRICS, ensure_ascii=False),
            'max_points': 60,
        }
        task_state = _build_task_state(self.adapter, record)
        with self.assertRaises(ValueError):
            self.adapter.match_score('回答', '回答', '', task_state)


class TestPLawBenchSamples(unittest.TestCase):

    def test_prompt_uses_official_task_instructions(self):
        adapter = _build_adapter()
        sample = adapter.record_to_sample({
            'id': 'case_analysis-1',
            'task': 'case_analysis',
            'judge_type': 'case_analysis',
            'category': '刑事实务',
            'context': '案情描述',
            'question': '争议问题',
            'rubrics': json.dumps(CASE_ANALYSIS_RUBRICS, ensure_ascii=False),
            'max_points': 60,
        })

        self.assertIsInstance(sample, Sample)
        self.assertEqual(sample.subset_key, 'case_analysis')
        self.assertIn('【法条依据】', sample.input)
        self.assertIn('## 案例\n案情描述', sample.input)
        self.assertIn('## 问题\n争议问题', sample.input)
        self.assertEqual(sample.metadata['max_points'], 60)


if __name__ == '__main__':
    unittest.main()
