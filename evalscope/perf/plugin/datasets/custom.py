from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            prompt = item.strip()
            is_valid, _ = self.check_prompt_length(prompt)
            if is_valid:
                if self.query_parameters.apply_chat_template:
                    message = self.create_message(prompt)
                    yield [message]
                else:
                    yield prompt


if __name__ == '__main__':
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark

    args = Arguments(
        model='qwen2.5-7b-instruct',
        url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        dataset_path='outputs/perf_data.txt',
        api_key='EMPTY',
        dataset='custom',
    )

    run_perf_benchmark(args)
