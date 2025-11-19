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
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    'background_change', 'color_alter', 'material_alter', 'motion_change', 'ps_human', 'style_change', 'subject-add',
    'subject-remove', 'subject-replace', 'text_change', 'tone_transfer'
]

LANGUAGE_LIST = ['en', 'cn']


@register_benchmark(
    BenchmarkMeta(
        name='gedit',
        pretty_name='GEdit-Bench',
        dataset_id='stepfun-ai/GEdit-Bench',
        description='GEdit-Bench Image Editing Benchmark, grounded in real-world '
        'usages is developed to support more authentic and '
        'comprehensive evaluation of image editing models.',
        tags=[Tags.IMAGE_EDITING],
        subset_list=SUBSET_LIST,
        metric_list=['Semantic Consistency', 'Perceptual Similarity'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        extra_params={
            'language': {
                'type': 'str',
                'description': f'Language of the instruction. Choices: {LANGUAGE_LIST}.',
                'value': 'en',
                'choices': LANGUAGE_LIST
            }
        }
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
        image_bytes = record['input_image']['bytes']
        input_image = bytes_to_base64(image_bytes, format='png', add_header=True)
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

        metadata = task_state.metadata
        text_prompt = metadata['instruction']
        input_image = metadata['input_image']  # base64 image
        edited_image = metadata[FileConstants.IMAGE_PATH]  # local image path
        _SC_prompt = self.SC_prompt.replace('<instruction>', text_prompt)

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=edited_image,
            prediction=edited_image,
        )

        # Build prompts
        SC_prompt_final = [
            ChatMessageUser(
                content=[
                    ContentImage(image=input_image),
                    ContentImage(image=edited_image),
                    ContentText(text=_SC_prompt)
                ]
            )
        ]
        PQ_prompt_final = [
            ChatMessageUser(content=[ContentImage(image=edited_image),
                                     ContentText(text=self.PQ_prompt)])
        ]

        guess_if_cannot_parse = True
        result_SC = self.llm_judge.judge(messages=SC_prompt_final)
        result_PQ = self.llm_judge.judge(messages=PQ_prompt_final)
        SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=guess_if_cannot_parse)
        PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=guess_if_cannot_parse)

        SC_score = min(SC_dict['score'])
        PQ_score = min(PQ_dict['score'])
        O_score = math.sqrt(SC_score * PQ_score)

        score.value = {'Semantic Consistency': SC_score, 'Perceptual Quality': PQ_score, 'Overall': O_score}
        score.main_score_name = 'Overall'
        score.metadata = {
            'SC_dict': SC_dict,
            'PQ_dict': PQ_dict,
        }
        return score
