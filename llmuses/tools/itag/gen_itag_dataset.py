# Copyright (c) Alibaba, Inc. and its affiliates.

import csv
import os

import json
import pandas as pd

from llmuses.utils.logger import get_logger

logger = get_logger()


class GenItagDataset:

    def __init__(self, template_file):
        self.csv_headers: list = []
        self._parse_template(template_file)

    def _parse_template(self, template_file) -> None:
        with open(template_file, 'r') as f:
            template_dict = json.load(f)
            field_group_list = template_dict['DataSource'][0]['Dataset']
            csv_headers = [item['FieldName'] for item in field_group_list]

        self.csv_headers = csv_headers

    def get_mock_data(self) -> list:
        # Generate mock data
        mock_data = list()
        mock_data.append(
            ['aaa' + str(i) for i in range(len(self.csv_headers))])
        mock_data.append(
            ['bbb' + str(i) for i in range(len(self.csv_headers))])
        mock_data.append(
            ['ccc' + str(i) for i in range(len(self.csv_headers))])
        mock_data.append(
            ['ddd' + str(i) for i in range(len(self.csv_headers))])

        return mock_data

    def gen_dataset(self, output_file):
        mock_data = self.get_mock_data()

        df = pd.DataFrame(mock_data, columns=self.csv_headers)
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

        logger.info(f'Generate dataset file: {output_file} successfully!')


if __name__ == '__main__':

    template_file_path = os.path.join(
        os.path.dirname('__file__'),
        'templates/template_xc_response_rank_0329.json')
    output_file_path = os.path.join(
        os.path.dirname('__file__'), 'datasets/llm_evals_datasets_rank.csv')

    gen_itag_dataset = GenItagDataset(template_file_path)
    gen_itag_dataset.gen_dataset(output_file_path)
