import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evalscope.agent.environments.local import TemporaryLocalAgentEnvironment
from evalscope.agent.tools.bash import BASH_TOOL_INFO
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.wide_search.utils import (
    METRIC_NAMES,
    WideSearchScorer,
    aggregate_official_scores,
    date_near,
    extract_number,
    number_near,
    url_match,
)
from evalscope.config import SandboxTaskConfig, TaskConfig
from evalscope.constants import JudgeStrategy


def _evaluation(metric: str = 'exact_match') -> dict:
    return {
        'unique_columns': ['id'],
        'required': ['id', 'value'],
        'eval_pipeline': {
            'id': {
                'preprocess': ['norm_str'],
                'metric': ['exact_match']
            },
            'value': {
                'preprocess': ['norm_str'],
                'metric': [metric],
                'criterion': 'The values must have the same meaning.',
            },
        },
    }


def _mapping_judge(prompt: str) -> str:
    if 'idx_0' in prompt:
        return '```json\n{"idx_0": 1}\n```'
    return '```json\n{}\n```'


class TestWideSearchScorer(unittest.TestCase):

    def setUp(self) -> None:
        self.scorer = WideSearchScorer(judge=_mapping_judge)
        self.gold = 'id,value\nA,one\nB,two\n'

    def test_perfect_markdown_table(self) -> None:
        prediction = '```markdown\n| id | value |\n| --- | --- |\n| A | one |\n| B | two |\n```'
        result = self.scorer.evaluate(prediction, self.gold, _evaluation())
        self.assertEqual(result.diagnostics['stage'], 'complete')
        self.assertEqual(result.values, {name: 1.0 for name in METRIC_NAMES})

    def test_bare_table_and_duplicate_key(self) -> None:
        prediction = '| id | value |\n| --- | --- |\n| A | one |\n| A | wrong |\n| B | two |'
        result = self.scorer.evaluate(prediction, self.gold, _evaluation())
        self.assertEqual(result.values['success_rate'], 1.0)
        self.assertEqual(result.diagnostics['prediction_rows'], 2)

    def test_missing_row_reduces_recall(self) -> None:
        prediction = '| id | value |\n| --- | --- |\n| A | one |'
        result = self.scorer.evaluate(prediction, self.gold, _evaluation())
        self.assertEqual(result.values['row_precision'], 1.0)
        self.assertEqual(result.values['row_recall'], 0.5)
        self.assertEqual(result.values['item_precision'], 1.0)
        self.assertEqual(result.values['item_recall'], 0.5)

    def test_extra_row_reduces_precision(self) -> None:
        prediction = '| id | value |\n| --- | --- |\n| A | one |\n| B | two |\n| C | three |'
        result = self.scorer.evaluate(prediction, self.gold, _evaluation())
        self.assertEqual(result.values['row_precision'], 2 / 3)
        self.assertEqual(result.values['row_recall'], 1.0)
        self.assertEqual(result.values['item_precision'], 2 / 3)
        self.assertEqual(result.values['item_recall'], 1.0)

    def test_wrong_cell_differs_at_row_and_item_levels(self) -> None:
        prediction = '| id | value |\n| --- | --- |\n| A | wrong |\n| B | two |'
        result = self.scorer.evaluate(prediction, self.gold, _evaluation())
        self.assertEqual(result.values['row_precision'], 0.5)
        self.assertEqual(result.values['item_precision'], 0.75)

    def test_invalid_table_returns_zero_scores(self) -> None:
        result = self.scorer.evaluate('not a table', self.gold, _evaluation())
        self.assertEqual(result.values, {name: 0.0 for name in METRIC_NAMES})
        self.assertEqual(result.diagnostics['stage'], 'parse')

    def test_llm_judge_scores_a_column(self) -> None:
        prediction = '| id | value |\n| --- | --- |\n| A | equivalent |'
        result = self.scorer.evaluate(prediction, 'id,value\nA,reference\n', _evaluation('llm_judge'))
        self.assertEqual(result.values['success_rate'], 1.0)
        self.assertIn('value', result.diagnostics['column_judges'])

    def test_semantic_column_mapping(self) -> None:

        def judge(prompt: str) -> str:
            if "['identifier', 'value']" in prompt:
                return '```json\n{"identifier": "id"}\n```'
            return '```json\n{}\n```'

        scorer = WideSearchScorer(judge=judge)
        prediction = '| identifier | value |\n| --- | --- |\n| A | one |\n| B | two |'
        result = scorer.evaluate(prediction, self.gold, _evaluation())
        self.assertEqual(result.values['success_rate'], 1.0)
        self.assertEqual(result.diagnostics['column_map'], {'identifier': 'id'})

    def test_invalid_llm_judge_response_scores_zero(self) -> None:
        scorer = WideSearchScorer(judge=lambda _: 'invalid')
        prediction = '| id | value |\n| --- | --- |\n| A | equivalent |'
        result = scorer.evaluate(prediction, 'id,value\nA,reference\n', _evaluation('llm_judge'))
        self.assertEqual(result.values['row_f1'], 0.0)
        self.assertEqual(result.values['item_f1'], 0.5)

    def test_official_number_date_and_url_boundaries(self) -> None:
        self.assertEqual(extract_number('about 1,234.5 kg'), '1234.5')
        self.assertEqual(number_near('101', '100', 0.01), 1.0)
        self.assertEqual(number_near('102', '100', 0.01), 0.0)
        self.assertEqual(date_near('2025-02-01', '2025-01-01'), 1.0)
        self.assertEqual(date_near('not a date', 'also invalid'), 1.0)
        self.assertEqual(url_match('https://example.com/a', 'http://example.com/b'), 1.0)

    def test_missing_judge_idx_scores_only_that_cell_zero(self) -> None:

        def judge(prompt: str) -> str:
            if 'idx_0' in prompt:
                return '```json\n{"idx_0": 1}\n```'
            return '```json\n{}\n```'

        scorer = WideSearchScorer(judge=judge)
        prediction = '| id | value |\n| --- | --- |\n| A | x |\n| B | y |'
        result = scorer.evaluate(prediction, 'id,value\nA,a\nB,b\n', _evaluation('llm_judge'))
        self.assertEqual(result.values['row_recall'], 0.5)
        self.assertEqual(result.values['item_recall'], 0.75)

    def test_judge_exception_scores_column_cells_zero(self) -> None:

        def judge(prompt: str) -> str:
            if 'idx_0' in prompt:
                raise RuntimeError('judge unavailable')
            return '```json\n{}\n```'

        scorer = WideSearchScorer(judge=judge)
        prediction = '| id | value |\n| --- | --- |\n| A | x |'
        result = scorer.evaluate(prediction, 'id,value\nA,a\n', _evaluation('llm_judge'))
        self.assertEqual(result.values['row_f1'], 0.0)
        self.assertEqual(result.values['item_f1'], 0.5)
        self.assertIn('judge unavailable', result.diagnostics['column_judges']['value'])


class TestWideSearchAggregation(unittest.TestCase):

    @staticmethod
    def _sample_score(group_id: int, language: str, success: float, row_f1: float) -> SampleScore:
        values = {name: row_f1 for name in METRIC_NAMES}
        values['success_rate'] = success
        return SampleScore(
            score=Score(value=values, main_score_name='success_rate'),
            sample_id=f'{language}-{group_id}',
            group_id=group_id,
            sample_metadata={'language': language},
        )

    def test_official_avg_pass_and_max_at_four(self) -> None:
        scores = []
        for value in [0.0, 1.0, 0.0, 0.0]:
            scores.append(self._sample_score(0, 'en', value, value * 0.8))
        for value in [0.0, 0.0, 0.0, 0.0]:
            scores.append(self._sample_score(1, 'zh', value, 0.2))
        aggregated = {(score.metric_name, score.aggregation_name): score.score
                      for score in aggregate_official_scores(scores)}
        self.assertEqual(aggregated[('all/success_rate', 'avg@4')], 0.125)
        self.assertEqual(aggregated[('all/success_rate', 'pass@4')], 0.5)
        self.assertEqual(aggregated[('en/row_f1', 'max@4')], 0.8)
        self.assertEqual(aggregated[('zh/row_f1', 'avg@4')], 0.2)

    def test_rejects_incomplete_repeat_groups(self) -> None:
        scores = [
            self._sample_score(0, 'en', 1.0, 1.0),
            self._sample_score(0, 'en', 1.0, 1.0),
            self._sample_score(1, 'zh', 1.0, 1.0),
        ]
        with self.assertRaisesRegex(ValueError, 'same number of trials'):
            aggregate_official_scores(scores)


class TestWideSearchAdapter(unittest.TestCase):

    @staticmethod
    def _write_dataset(root: Path, count: int = 2) -> None:
        (root / 'widesearch_gold').mkdir()
        records = []
        for index in range(count):
            language = 'en' if index < count // 2 else 'zh'
            instance_id = f'ws_{language}_{index + 1:03d}'
            records.append({
                'instance_id': instance_id,
                'query': f'query-{language}-{index}',
                'evaluation': json.dumps(_evaluation()),
                'language': language,
            })
            (root / 'widesearch_gold' / f'{instance_id}.csv').write_text('id,value\nA,one\n', encoding='utf-8')
        (root / 'widesearch.jsonl').write_text('\n'.join(json.dumps(record) for record in records), encoding='utf-8')

    def test_local_dataset_load_and_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_dataset(root)
            config = TaskConfig(
                model='mock',
                datasets=['wide_search'],
                repeats=2,
                dataset_args={'wide_search': {
                    'local_path': tmp_dir
                }},
                judge_model_args={'model_id': 'mock'},
            )
            adapter = get_benchmark('wide_search', config=config)
            dataset = adapter.load_dataset()['default']
            self.assertEqual(len(dataset), 4)
            self.assertEqual([sample.group_id for sample in dataset], [0, 0, 1, 1])
            self.assertEqual([sample.metadata['language'] for sample in dataset], ['en', 'en', 'zh', 'zh'])
            self.assertTrue(all(sample.target.startswith('id,value') for sample in dataset))

    def test_remote_snapshot_full_distribution_and_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_dataset(root, count=200)
            config = TaskConfig(
                model='mock',
                datasets=['wide_search'],
                judge_model_args={'model_id': 'mock'},
            )
            with patch('modelscope.dataset_snapshot_download', return_value=tmp_dir):
                dataset = get_benchmark('wide_search', config=config).load_dataset()['default']
            self.assertEqual(len(dataset), 200)
            self.assertEqual(sum(sample.metadata['language'] == 'en' for sample in dataset), 100)
            self.assertEqual(sum(sample.metadata['language'] == 'zh' for sample in dataset), 100)
            self.assertTrue(all(sample.target == 'id,value\nA,one\n' for sample in dataset))

    def test_local_environment_is_cleaned(self) -> None:
        environment = TemporaryLocalAgentEnvironment('sample')
        working_dir = environment.working_dir
        self.assertTrue(working_dir.exists())
        import asyncio
        result = asyncio.run(environment.exec(['/bin/bash', '-c', 'pwd && touch executed.txt']))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(Path(result.stdout.strip()).resolve(), working_dir.resolve())
        self.assertTrue((working_dir / 'executed.txt').exists())
        asyncio.run(environment.close())
        self.assertFalse(working_dir.exists())

    def test_rule_only_judge_is_rejected(self) -> None:
        config = TaskConfig(
            model='mock',
            datasets=['wide_search'],
            judge_strategy=JudgeStrategy.RULE,
            judge_model_args={'model_id': 'mock'},
        )
        adapter = get_benchmark('wide_search', config=config)
        with self.assertRaisesRegex(ValueError, "judge_strategy='auto' or 'llm'"):
            adapter._validate_judge_config()

    def test_base_metric_pipeline_uses_official_scorer(self) -> None:
        config = TaskConfig(
            model='mock',
            datasets=['wide_search'],
            judge_strategy=JudgeStrategy.LLM,
            judge_model_args={'model_id': 'mock'},
        )
        adapter = get_benchmark('wide_search', config=config)
        adapter.llm_judge = Mock(judge=_mapping_judge)
        sample = Sample(
            id=3,
            group_id=2,
            input='question',
            target='id,value\nA,one\n',
            metadata={
                'instance_id': 'ws_en_001',
                'language': 'en',
                'evaluation': _evaluation(),
            },
        )
        task_state = TaskState(
            model='mock',
            sample=sample,
            output=ModelOutput.from_content(model='mock', content='| id | value |\n| --- | --- |\n| A | one |'),
            completed=True,
        )

        sample_score = adapter.calculate_metrics(task_state)

        self.assertEqual(sample_score.sample_id, 3)
        self.assertEqual(sample_score.group_id, 2)
        self.assertEqual(sample_score.score.value, {name: 1.0 for name in METRIC_NAMES})

    def test_docker_uses_unified_sandbox_and_agent_timeout(self) -> None:
        config = TaskConfig(
            model='mock',
            datasets=['wide_search'],
            agent_config=NativeAgentConfig(command_timeout=17),
            sandbox=SandboxTaskConfig(
                enabled=True,
                default_config={
                    'image': 'custom:latest',
                    'network_enabled': False,
                },
            ),
            judge_model_args={'model_id': 'mock'},
        )
        adapter = get_benchmark('wide_search', config=config)
        sample = Sample(input='question', metadata={'instance_id': 'docker-test', 'language': 'en'})
        with patch('evalscope.benchmarks.wide_search.wide_search_adapter.check_import'
                   ), patch('evalscope.agent.environments.enclave.EnclaveAgentEnvironment') as environment_cls:
            adapter.build_environment(sample)
        environment_cls.assert_called_once_with(
            engine='docker',
            sandbox_config={
                'image': 'custom:latest',
                'network_enabled': False,
            },
            timeout=17,
        )

    def test_official_prompts_function_calling_and_max_steps(self) -> None:
        config = TaskConfig(
            model='mock',
            datasets=['wide_search'],
            agent_config=NativeAgentConfig(max_steps=7, command_timeout=12),
            judge_model_args={'model_id': 'mock'},
        )
        adapter = get_benchmark('wide_search', config=config)
        sample = Sample(
            input='question', tools=[BASH_TOOL_INFO], metadata={
                'instance_id': 'ws_zh_001',
                'language': 'zh'
            }
        )
        self.assertEqual(adapter.build_strategy(sample).name, 'function_calling')
        self.assertEqual(adapter._resolve_max_steps(config.agent_config), 7)
        self.assertIn('联网信息搜索专家', adapter.build_initial_messages(sample)[0].content)
        handlers, tools = adapter._resolve_tools(sample, config.agent_config)
        self.assertIn('bash', handlers)
        self.assertEqual(tools[0].parameters.properties['timeout'].default, 12)


if __name__ == '__main__':
    unittest.main()
