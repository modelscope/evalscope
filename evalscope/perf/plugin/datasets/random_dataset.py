import numpy as np
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('random')
class RandomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        assert self.query_parameters.tokenizer_path, 'Tokenizer path is required for random data generation, please provide it with `--tokenizer_path`.'  # noqa: E501

        from modelscope import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.query_parameters.tokenizer_path, trust_remote_code=True)
        self.prefix_length = self.query_parameters.prefix_length
        self.prefix_ids = self.get_random_inputs(self.prefix_length)
        self.template_len = self.get_template_len()
        self.number = self.query_parameters.number or 1

    def build_messages(self) -> Iterator[List[Dict]]:
        if self.query_parameters.apply_chat_template:
            min_prompt_length = self.query_parameters.min_prompt_length - self.template_len
            max_prompt_length = self.query_parameters.max_prompt_length - self.template_len + 1
        else:
            min_prompt_length = self.query_parameters.min_prompt_length
            max_prompt_length = self.query_parameters.max_prompt_length + 1

        assert min_prompt_length >= 0, f'min_prompt_length should be greater than or equal to the template length {self.template_len}.'  # noqa: E501
        assert max_prompt_length >= min_prompt_length, 'max_prompt_length should be greater than or equal to min_prompt_length.'  # noqa: E501

        # refer to https://github.com/vllm-project/vllm/blob/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/benchmarks/benchmark_serving.py#L366C1-L399C1  # noqa: E501
        input_lens = np.random.randint(min_prompt_length, max_prompt_length, size=self.number)
        offsets = np.random.randint(0, self.tokenizer.vocab_size, size=self.number)

        for i in range(self.number):
            prompt_ids = ((offsets[i] + i + np.arange(input_lens[i])) % self.tokenizer.vocab_size).tolist()
            prompt = self.tokenizer.decode(self.prefix_ids + prompt_ids)

            if self.query_parameters.apply_chat_template:
                yield [{'role': 'user', 'content': prompt}]
            else:
                yield prompt

    def get_random_inputs(self, length: int) -> List[int]:
        if length <= 0:
            return []
        input_ids = np.random.randint(0, self.tokenizer.vocab_size, size=length).tolist()
        return input_ids

    def get_template_len(self):
        empty_message = [{'role': 'user', 'content': ''}]
        template = self.tokenizer.apply_chat_template(empty_message, tokenize=True, add_generation_prompt=True)
        return len(template)
