# Copyright (c) Alibaba, Inc. and its affiliates.

import pandas as pd
from odps import ODPS


class MaxComputeUtil:
    """
    MaxCompute util class.

    Args:
        access_id: your access id of MaxCompute
        access_key: access key of MaxCompute
        project_name: your project name of MaxCompute
        endpoint: endpoint of MaxCompute

    Attributes:
        _odps: ODPS object

    """

    def __init__(self, access_id, access_key, project_name, endpoint):
        self._odps = ODPS(access_id, access_key, project_name, endpoint)

    def get_table(self, table_name):
        """
        Get MaxCompute table object.
        """
        return self._odps.get_table(table_name)

    def read_data(self, table_name: str, pt_condition: str) -> pd.DataFrame:
        """
        Read data from MaxCompute table.
        :param table_name: table name
        :param pt_condition: partition condition,
            Example: pt_condition = 'dt=20230331'
        :return: pandas dataframe with all data
        """
        t = self.get_table(table_name)

        with t.open_reader(partition=pt_condition) as reader:
            pd_df = reader.to_pandas()

        return pd_df

    def fetch_data(self, table_name: str, pt_condition: str,
                   output_path: str) -> None:
        """
        Fetch data from MaxCompute table to local file.
        :param table_name: table name
        :param pt_condition: partition condition,
            Example: pt_condition = 'dt=20230331'
        :param output_path: output path
        :return: None
        """
        pd_df = self.read_data(table_name, pt_condition)
        pd_df.to_csv(output_path, index=False)
        print(f'Fetch data to {output_path} successfully.')
