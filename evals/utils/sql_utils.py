# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from simple_ddl_parser import parse_from_file


def get_table_schema(ddl_file) -> dict:
    """
    Get table schema from DDL file.
    :param ddl_file:  local DDL file path.
    :return:  a dict which contains table schema. e.g.:
        {'columns': [{'name': 'gmt_create'}, {...}], 'primary_key': [],
            'partitioned_by': {}, 'table_name': 'itag_task',
            'if_not_exists': true, 'comment': 'xxx', 'LIFECYCLE': 7}
    """
    if not os.path.isfile(ddl_file):
        raise FileNotFoundError(f'DDL file not found: {ddl_file}')

    ddl_result = parse_from_file(ddl_file)
    if not ddl_result:
        raise ValueError(f'Parse DDL result is empty: {ddl_file}')

    return ddl_result[0]
