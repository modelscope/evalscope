# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from typing import Any

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.acebench.acebench_adapter import AceBenchAdapter


class TestAceBenchAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AceBenchAdapter(benchmark_meta=BENCHMARK_REGISTRY['acebench'], task_config=None)

    def test_record_to_sample_normalizes_tools_and_messages(self) -> None:
        record = {
            'id': 'normal_single_turn_single_function_0',
            'sub_category': 'data_normal_single_turn_single_function',
            'question': 'user: Book a trip.\nsystem: Which destination?\nuser: Paris.',
            'function': (
                '[{"name":"BookTrip","description":"Book travel.","parameters":'
                '{"type":"dict","required":["destination"],"properties":'
                '{"destination":{"type":"string"}}}}]'
            ),
            'rubric': '{"ground_truth":{"BookTrip":{"destination":"Paris"}}}',
        }

        sample = self.adapter.record_to_sample(record)

        self.assertEqual(sample.tools[0].parameters.type, 'object')
        self.assertEqual(sample.input[1].role, 'user')
        self.assertEqual(sample.input[2].role, 'assistant')
        self.assertEqual(sample.metadata['ground_truth'], {'BookTrip': {'destination': 'Paris'}})

    def test_match_score_text_function_call(self) -> None:
        sample = self._sample({'BookTrip': {'destination': 'Paris', 'passengers': 2}})
        prediction = "[BookTrip(destination='Paris', passengers=2)]"
        state = TaskState(
            model='mock',
            sample=sample,
            output=ModelOutput.from_content(model='mock', content=prediction),
            completed=True,
        )

        score = self.adapter.match_score(prediction, prediction, sample.target, state)

        self.assertEqual(score.value['acc'], 1.0)

    def test_match_score_native_tool_call(self) -> None:
        sample = self._sample({'BookTrip': {'destination': 'Paris', 'passengers': 2}})
        output = ModelOutput.for_tool_call(
            model='mock',
            tool_name='BookTrip',
            tool_arguments={'destination': 'Paris', 'passengers': 2},
        )
        state = TaskState(model='mock', sample=sample, output=output, completed=True)

        score = self.adapter.match_score('', '', sample.target, state)

        self.assertEqual(score.value['acc'], 1.0)

    def test_match_score_special_incomplete(self) -> None:
        sample = self._sample(
            {'BookTrip': ['destination']},
            sub_category='data_special_incomplete',
        )
        prediction = '["Missing necessary parameters (destination) for the api (BookTrip)"]'
        state = TaskState(
            model='mock',
            sample=sample,
            output=ModelOutput.from_content(model='mock', content=prediction),
            completed=True,
        )

        score = self.adapter.match_score(prediction, prediction, sample.target, state)

        self.assertEqual(score.value['acc'], 1.0)

    def test_match_score_agent_process(self) -> None:
        sample = self._sample(
            [{'BaseApi': {'wifi': True}}],
            sub_category='data_agent_multi_turn',
            mile_stone=['[turn_on_wifi()]', "[send_message(sender_name='Grace', receiver_name='Frank')]"],
        )
        prediction = "[turn_on_wifi(), send_message(sender_name='Grace', receiver_name='Frank')]"
        state = TaskState(
            model='mock',
            sample=sample,
            output=ModelOutput.from_content(model='mock', content=prediction),
            completed=True,
        )

        score = self.adapter.match_score(prediction, prediction, sample.target, state)

        self.assertEqual(score.value['acc'], 1.0)
        self.assertEqual(score.value['process_acc'], 1.0)

    @staticmethod
    def _sample(
        ground_truth: Any,
        sub_category: str = 'data_normal_single_turn_single_function',
        mile_stone: Any = None,
    ) -> Sample:
        return Sample(
            input='question',
            target='',
            metadata={
                'sub_category': sub_category,
                'ground_truth': ground_truth,
                'mile_stone': mile_stone or [],
            },
        )


if __name__ == '__main__':
    unittest.main()
