# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, ImageEditAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessage, ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import FileConstants, Tags
from evalscope.utils.io_utils import PIL_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    'background_change', 'color_alter', 'material_alter', 'motion_change', 'ps_human', 'style_change', 'subject-add',
    'subject-remove', 'subject-replace', 'text_change', 'tone_transfer'
]


@register_benchmark(
    BenchmarkMeta(
        name='gedit',
        pretty_name='GEdit-Bench',
        dataset_id='stepfun-ai/GEdit-Bench',
        description='GEdit-Bench Image Editing Benchmark',
        tags=[Tags.IMAGE_EDITING],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        extra_params={'language': '[choose `en` or `zh`, default `en`]'}
    )
)
class GEditAdapter(ImageEditAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.language = self.extra_params.get('language', 'en')
        self.reformat_subset = True
        self._use_llm_judge = True

        self.load_prompt()

    def load_prompt(self):
        from . import vie_prompts

        self.context = vie_prompts._context_no_delimit
        self.SC_prompt = '\n'.join([
            self.context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC
        ])
        self.PQ_prompt = '\n'.join([self.context, vie_prompts._prompts_0shot_rule_PQ])

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        record = copy.deepcopy(record)

        # Process instruction and image
        instruction = record['instruction']
        input_image = PIL_to_base64(record['input_image'])
        record['input_image'] = input_image
        record[FileConstants.ID] = record['key']
        del record['input_image_raw']

        text_content = ContentText(text=instruction)
        image_content = ContentImage(image=input_image)

        messages: List[ChatMessage] = [
            ChatMessageUser(content=[text_content, image_content]),
        ]

        return Sample(input=messages, subset_key=record['task_type'], metadata=record)

    def sample_filter(self, sample: Sample) -> bool:
        language = sample.metadata.get('instruction_language', 'en')
        return super().sample_filter(sample) and language == self.language

    def llm_match_score(self, original_prediction, filtered_prediction, reference, task_state: TaskState) -> Score:
        import math

        from .utils import mllm_output_to_dict

        _SC_prompt = self.SC_prompt.replace('<instruction>', text_prompt)
        SC_prompt_final = self.model.prepare_prompt(image_prompts, _SC_prompt)
        PQ_prompt_final = self.model.prepare_prompt(image_prompts[-1], self.PQ_prompt)

        results_dict = {}

        SC_dict = False
        PQ_dict = False
        tries = 0
        max_tries = 1
        while SC_dict is False or PQ_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            result_SC = self.llm_judge.judge(SC_prompt_final)
            result_PQ = self.llm_judge.judge(PQ_prompt_final)
            SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=guess_if_cannot_parse)
            PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=guess_if_cannot_parse)

        SC_score = min(results_dict['SC']['score'])
        PQ_score = min(results_dict['PQ']['score'])
        O_score = math.sqrt(SC_score * PQ_score)

        results_dict['SC'] = SC_dict
        results_dict['PQ'] = PQ_dict
        results_dict['O'] = O_score
        return results_dict
