# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been adapted from the original version of the SciCode benchmark.
# Modifications have been made by xantheocracy, 2024.
# Original source: https://github.com/scicode-bench/SciCode/blob/main/src/scicode/parse/parse.py

import scipy  # type: ignore
from h5py import Dataset, File, Group  # type: ignore
from typing import Any, TypeAlias

H5PY_FILE = '/workspace/test_data.h5'

SparseMatrix: TypeAlias = (scipy.sparse.coo_matrix | scipy.sparse.bsr_matrix | scipy.sparse.csr_matrix)


def process_hdf5_list(group: Group) -> list[Any]:
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group: Group) -> dict[str | float, Any]:
    dict = {}
    for key, obj in group.items():
        if isinstance(obj, Group):
            dict[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group: Group) -> SparseMatrix:
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group: Group, ) -> list[Any] | dict[str | float, Any] | SparseMatrix:
    assert len(group) > 0
    for key in group.keys():
        if key == 'list':
            return process_hdf5_list(group[key])
        if key == 'sparse_matrix':
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)
    raise ValueError('No valid key found in the group to process')


def process_hdf5_to_tuple(step_id: str, test_num: int) -> list[Any]:
    data_lst: list[Any] = []
    with File(H5PY_FILE, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], Group):
                group = f[group_path]  # test1, test2, test3
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:  # only 1 var in the test
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst: list[Any] = []
                    for key in group.keys():  # var1, var2, var3
                        subgroup = group[key]
                        if isinstance(subgroup, Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(f'Path {group_path} not found in the file.')
    return data_lst
