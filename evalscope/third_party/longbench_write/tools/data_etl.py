# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os.path
import re
from typing import List

from evalscope.third_party.longbench_write.eval import EvalLength
from evalscope.third_party.longbench_write.utils import chinese_to_arabic, count_words
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()
"""
This script is used to preprocess the dataset for the LongWriter.
"""


class DataETL:

    def __init__(self, in_file: str, out_dir: str):
        self.data: List[dict] = jsonl_to_list(in_file)
        logger.info(f'Loaded {len(self.data)} samples from {in_file}')

        self.out_dir = out_dir

    @staticmethod
    def filter(strategies: List[str], example: dict, verbose: bool = False) -> dict:
        # example = {'messages': [{'role': 'user', 'content': 'xxx'}, {'role': 'assistant', 'content': 'xxx'}]}

        for strategy in strategies:
            if strategy == 'no_required_length':
                pattern1 = r'(\d+)字'
                pattern2 = r'(.)百字|(.)千字|(.)万字'
                pattern3 = r'(\d+) words'

                text = example['messages'][0]['content']
                matches = re.findall(pattern1, text)
                matches.extend(re.findall(pattern2, text))
                matches.extend(re.findall(pattern3, text))
                matches = list(set(matches))

                final_matches = []
                for item in matches:
                    if isinstance(item, tuple):
                        # Note: refer to pattern2
                        for idx in range(len(item)):
                            num = chinese_to_arabic(item[idx])
                            if idx == 0 and item[idx]:
                                if isinstance(num, int):
                                    final_matches.append(num * 100)
                            elif idx == 1 and item[idx]:
                                if isinstance(num, int):
                                    final_matches.append(num * 1000)
                            elif idx == 2 and item[idx]:
                                if isinstance(num, int):
                                    final_matches.append(num * 10000)
                            else:
                                continue
                    else:
                        final_matches.append(item)

                final_matches = list(set(final_matches))
                if verbose and len(final_matches) > 0:
                    logger.info(f'>>final_matches: {final_matches}')

                if len(final_matches) == 1:
                    required_length = int(final_matches[0])
                    example['length'] = required_length
                else:
                    example = {}
            else:
                raise ValueError(f'Unknown strategy: {strategy}')

        return example

    def process_filter(self):
        # Entry point for processing the data
        filtered_data = []
        for ex in self.data:
            filtered_ex = DataETL.filter(strategies=['no_required_length'], example=ex, verbose=True)
            if len(filtered_ex) > 0:
                filtered_data.append(filtered_ex)

        for example in filtered_data:
            assistant_messages = example['messages'][1]
            assert assistant_messages['role'] == 'assistant'
            response_length, _ = count_words(assistant_messages['content'])
            example['response_length'] = response_length

        # Dump the filtered data to the output file
        # example = {'messages': [{'role': 'user', 'content': 'xxx'}, {'role': 'assistant', 'content': 'xxx'}], 'length': 1000, 'response_length': 1000'}
        out_file = os.path.join(self.out_dir, 'long_filtered.jsonl')
        with open(out_file, 'w', encoding='utf-8') as f:
            for example in filtered_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        return out_file

    def process_eval_length(self, in_file: str, threshold: float = 80):
        data_list = jsonl_to_list(in_file)

        res_list = []
        x_list = []
        y_list = []
        x_filter_list = []
        y_filter_list = []
        for ex in data_list:
            x = ex['length']
            y = ex['response_length']
            if x == 0 or y == 0:
                continue

            if x / y > 3:
                print(f'\n>>ex: {json.dumps(ex, ensure_ascii=False)}')
                print(f'>>length: {x}, response_length: {y}\n')

            x_list.append(x)
            y_list.append(y)

            len_score = EvalLength.score(x, y)
            if len_score >= threshold:
                ex['score_length'] = len_score
                res_list.append(ex)

                x_filter_list.append(x)
                y_filter_list.append(y)

        out_file = os.path.join(self.out_dir, 'long_filtered_length.jsonl')
        logger.info(f'Got {len(res_list)} examples left')
        with open(out_file, 'w', encoding='utf-8') as f:
            for example in res_list:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        plt = EvalLength.plot_img(x_list, y_list)
        plt.savefig(os.path.join(self.out_dir, 'length_scatter.png'))

        plt_filter = EvalLength.plot_img(x_filter_list, y_filter_list)
        plt_filter.savefig(os.path.join(self.out_dir, 'length_scatter_filtered.png'))

        return out_file


if __name__ == '__main__':
    # run `no_required_length`: got 1748 exampels left

    # Refer to: https://modelscope.cn/datasets/ZhipuAI/LongWriter-6k/files
    longbench_write_file = '/path/to/long.jsonl'
    out_dir = '/path/to/your_output_dir'

    data_etl = DataETL(in_file=longbench_write_file, out_dir=out_dir)
    filtered_file_path = data_etl.process_filter()
    print(f'Filtered data is saved in {filtered_file_path}')

    eval_l_file_path = data_etl.process_eval_length(in_file=filtered_file_path, threshold=80)
