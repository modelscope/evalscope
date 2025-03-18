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
        self.prefix = self.get_random_inputs(self.prefix_length)
        self.template_len = self.get_template_len()
        self.number = self.query_parameters.number or 1

    def build_messages(self) -> Iterator[List[Dict]]:
        min_prompt_length = self.query_parameters.min_prompt_length - self.template_len
        max_prompt_length = self.query_parameters.max_prompt_length - self.template_len + 1

        assert min_prompt_length >= 0, f'min_prompt_length should be greater than or equal to the template length {self.template_len}.'  # noqa: E501
        assert max_prompt_length >= min_prompt_length, 'max_prompt_length should be greater than or equal to min_prompt_length.'  # noqa: E501

        for _ in range(self.number):
            prompt_length = np.random.randint(min_prompt_length, max_prompt_length)
            prompt = self.get_random_inputs(prompt_length)
            prompt_str = self.tokenizer.decode(self.prefix + prompt, skip_special_tokens=False)
            yield [{'role': 'user', 'content': prompt_str}]

    def get_random_inputs(self, length: int) -> List[int]:
        if length <= 0:
            return []
        input_ids = np.random.randint(0, self.tokenizer.vocab_size, size=length).tolist()
        return input_ids

    def get_template_len(self):
        empty_message = [{'role': 'user', 'content': ''}]
        template = self.tokenizer.apply_chat_template(empty_message, tokenize=True, add_generation_prompt=True)
        return len(template)
