from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentAudio, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import prompt

logger = get_logger()

MULT_CHOICE_PROMPT = r"""
Answer the following multiple choice question based on the image and audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='omni_bench',
        pretty_name='OmniBench',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=
        'OmniBench, a pioneering universal multimodal benchmark designed to rigorously evaluate MLLMs\' capability to recognize, interpret, and reason across visual, acoustic, and textual inputs simultaneously.',  # noqa: E501
        dataset_id='m-a-p/OmniBench',
        metric_list=['acc'],
        eval_split='train',
        prompt_template=MULT_CHOICE_PROMPT,
        extra_params={
            'use_image': {
                'type': 'bool',
                'description': 'Whether to provide the raw image. False uses textual alternative.',
                'value': True
            },
            'use_audio': {
                'type': 'bool',
                'description': 'Whether to provide the raw audio. False uses textual alternative.',
                'value': True
            }
        }
    )
)
class OmniBenchAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.use_image = self.extra_params.get('use_image', True)
        self.use_audio = self.extra_params.get('use_audio', True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        options = record['options']
        answer = record['answer']
        answer_char = chr(ord('A') + options.index(answer))

        input_text = prompt(question=question, choices=options, template=MULT_CHOICE_PROMPT)
        content_list: List[Content] = [ContentText(text=input_text)]

        if self.use_image:
            image_base64 = bytes_to_base64(record['image']['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        else:
            alt_image = record['image content']
            content_list.append(ContentText(text=f'[Image Alternative Text]: {alt_image}'))

        if self.use_audio:
            audio_base64 = bytes_to_base64(
                record['audio']['bytes'], format='mp3', add_header=True, content_type='audio'
            )
            content_list.append(ContentAudio(audio=audio_base64, format='mp3'))
        else:
            alt_audio = record['audio content']
            content_list.append(ContentText(text=f'[Audio Alternative Text]: {alt_audio}'))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=options,
            target=answer_char,
            metadata={
                'index': record['index'],
                'task_type': record['task type'],
                'audio_type': record['audio type'],
                'answer': answer,
                'image_content': record['image content'],
                'audio_content': record['audio content'],
            }
        )
