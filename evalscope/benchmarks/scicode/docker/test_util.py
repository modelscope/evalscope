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
# Modifications have been made by xantheocracy, 2025.
# Original source: https://github.com/scicode-bench/SciCode/blob/main/src/scicode/compare/cmp.py

from typing import Any

import numpy as np
import scipy.sparse  # type: ignore
import sympy  # type: ignore


def are_dicts_close(
    dict1: dict[Any, Any], dict2: dict[Any, Any], atol: float = 1e-8, rtol: float = 1e-5
) -> bool:
    dict1 = process_symbol_in_dict(dict1)
    dict2 = process_symbol_in_dict(dict2)
    # Check if both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        return False

    # Check if the corresponding values are close
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, sympy.Symbol | str):
            if not value1 == value2:
                return False
        elif isinstance(
            value1,
            scipy.sparse.csr_matrix
            | scipy.sparse.csc_matrix
            | scipy.sparse.bsr_matrix
            | scipy.sparse.coo_matrix,
        ):
            value1 = value1.toarray()
            value2 = value2.toarray()
            if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                return False
        # Use np.allclose to compare values
        else:
            try:
                if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                    return False
            except ValueError:
                if not value1 == value2:
                    return False

    return True


def process_symbol_in_dict(dict: dict[Any, Any]) -> dict[Any, Any]:
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value
        if isinstance(value, sympy.Symbol):
            new_dict[key] = str(value)
        if isinstance(key, sympy.Symbol):
            new_dict[str(key)] = dict[key]
            new_dict.pop(key)
    return new_dict


def are_csc_matrix_close(
    matrix1: scipy.sparse.csr_matrix | scipy.sparse.csc_matrix, matrix2: Any
) -> bool:
    dense1 = matrix1.toarray()
    dense2 = matrix2.toarray()
    return np.allclose(dense1, dense2)


def cmp_tuple_or_list(var1: list[Any], var2: list[Any]) -> bool:
    if len(var1) != len(var2):
        return False
    for v1, v2 in zip(var1, var2):
        if isinstance(v1, dict):
            if not are_dicts_close(v1, v2):
                return False
        elif isinstance(v1, scipy.sparse.csr_matrix | scipy.sparse.csc_matrix):
            if not are_csc_matrix_close(v1, v2):
                return False
        elif isinstance(v1, bool):
            if not v1 == v2:
                return False
        else:
            try:
                if not np.allclose(v1, v2):
                    return False
            except ValueError as e:
                print(e)
                if not v1 == v2:
                    return False
    return True
