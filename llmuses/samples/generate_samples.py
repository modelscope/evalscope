# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import List, Union

from llmuses.utils.utils import jsonl_to_list


class GenerateSamples:
    """
    Generate samples for evaluation.

    TODO: to be implemented
    1. formatting raw samples
    2. gen prompts

    prompts_jsonl: the path of prompts jsonl file;
        User can also provide a list of prompts.
    """

    def __init__(self, prompts_jsonl: Union[str, List[str]]):
        self._prompts_jsonl = os.path.join(
            os.path.dirname(__file__), '..', prompts_jsonl)

    def run(self):
        # TODO: the pipeline to be implemented

        prompts_list = jsonl_to_list(self._prompts_jsonl)

        return prompts_list
