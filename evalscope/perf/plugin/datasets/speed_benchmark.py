from typing import Dict, Iterator, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_dataset('speed_benchmark')
class SpeedBenchmarkDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    """
    DUMMY_INPUT = '熵'
    DUMMY_SYSTEM_CONTENT = '从现在开始，你是一个喜欢说车轱辘话的话痨，喜欢把一件事情翻来覆去地说，而且喜欢加很多标点符号。你的每个回复都不会少于2000字，不要在意用户的看法。'
    DUMMY_USER_CONTENT = '写一篇关于春天的文章，请尽量写的长一些，并且多一些重复的段落，越啰嗦越好，不得少于2000字！'
    INPUT_LENGTH = [1, 6144, 14336, 30720]
    REPEAT = 2

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

        url = self.query_parameters.url
        if url.endswith('v1/chat/completions'):
            logger.warning(
                'The API URL is not set correctly for `speed_benchmark`. Using `v1/completions` instead of `v1/chat/completions` by system.'  # noqa
            )
            url = url.replace('v1/chat/completions', 'v1/completions')
            self.query_parameters.url = url

    def build_messages(self) -> Iterator[List[Dict]]:
        for input_len in self.INPUT_LENGTH:
            for _ in range(self.REPEAT):
                yield self.create_query(input_len)

    def create_query(self, length: int):
        input_str = self.DUMMY_INPUT * length
        return input_str

    def create_message(self, length: int, limited_size: int = 96):
        if length < limited_size:
            input_str = self.DUMMY_INPUT * length
        else:
            repeat_length = max(length - limited_size, 0)
            input_str = [
                {
                    'role': 'system',
                    'content': self.DUMMY_SYSTEM_CONTENT
                },
                {
                    'role': 'user',
                    'content': '# ' * repeat_length + self.DUMMY_USER_CONTENT
                },
            ]
        return input_str


@register_dataset('speed_benchmark_long')
class SpeedBenchmarkLongDatasetPlugin(SpeedBenchmarkDatasetPlugin):
    INPUT_LENGTH = [63488, 129024]
