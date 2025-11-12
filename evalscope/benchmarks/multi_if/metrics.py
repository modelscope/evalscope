# Copyright (c) Meta Platforms, Inc. and affiliates.
#
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
from typing import Any, Dict, List, Optional, Tuple

from evalscope.utils import get_logger
from . import ifeval

logger = get_logger()


def gen_acc_strict(x: Dict[str, Any]) -> Dict[str, List]:
    # reference: fbcode/gen_ai/github/fair_evals/evals/tasks/finetune/ifeval.py
    response = str(x['response'])
    instruction_list = x['instruction_id_list']
    is_following_list = []
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = ifeval.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**x['kwargs'][index])
        if response and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return {
        'follow_instruction_list': is_following_list,
        'instruction_id_list': instruction_list,
    }


def gen_acc_loose(x: Dict[str, Any]) -> Dict[str, List]:
    response = str(x['response'])
    r = response.split('\n')
    response_remove_first = '\n'.join(r[1:]).strip()
    response_remove_last = '\n'.join(r[:-1]).strip()
    response_remove_both = '\n'.join(r[1:-1]).strip()
    revised_response = response.replace('*', '')
    revised_response_remove_first = response_remove_first.replace('*', '')
    revised_response_remove_last = response_remove_last.replace('*', '')
    revised_response_remove_both = response_remove_both.replace('*', '')
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = x['instruction_id_list']
    is_following_list = []
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = ifeval.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**x['kwargs'][index])

        is_following = False
        for r in all_responses:  # type: ignore
            if r.strip() and instruction.check_following(r):  # type: ignore
                is_following = True
                break

        is_following_list.append(is_following)
    return {
        'follow_instruction_list': is_following_list,
        'instruction_id_list': instruction_list,
    }


def parse_result(outputs: List[Dict[str, Any]]) -> Tuple[float, float]:

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    for example in outputs:
        follow_instruction_list = example['follow_instruction_list']
        instruction_id_list = example['instruction_id_list']

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

    return prompt_correct / prompt_total if prompt_total > 0 else 0, \
        instruction_correct / instruction_total if instruction_total > 0 else 0


def parse_result_no_reduce(outputs: List[Dict[str, Any]]) -> Tuple[List, List]:

    prompt_res = []
    inst_res = []

    for example in outputs:
        follow_instruction_list = example['follow_instruction_list']
        instruction_id_list = example['instruction_id_list']
        if all(follow_instruction_list):
            prompt_res.append(1)
        else:
            prompt_res.append(0)
        inst_res.append(sum(follow_instruction_list) / len(instruction_id_list) if instruction_id_list else 0.0)

    return prompt_res, inst_res
