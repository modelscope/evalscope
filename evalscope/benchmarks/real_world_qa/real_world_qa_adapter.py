import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = ['default']

OPEN_PROMPT = (
    'Read the picture and solve the following problem step by step.'
    'The last line of your response should be of the form'
    ' "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.\n\n'
    '{question}\n\n'
    'Remember to put your answer on its own line at the end in the form'
    ' "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem,'
    ' and you do not need to use a \\boxed command.'
)


@register_benchmark(
    BenchmarkMeta(
        name='real_world_qa',
        pretty_name='RealWorldQA',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

RealWorldQA is a benchmark contributed by XAI designed to evaluate multimodal AI models' understanding of real-world spatial and physical environments. It uses authentic images from everyday scenarios to test practical visual comprehension.

## Task Description

- **Task Type**: Real-World Visual Question Answering
- **Input**: Real-world image with spatial/physical question
- **Output**: Verifiable answer about the scene
- **Domain**: Physical environments, driving scenarios, everyday scenes

## Key Features

- 700+ images from real-world scenarios
- Includes vehicle-captured images (driving scenes)
- Questions with verifiable ground-truth answers
- Tests spatial understanding and physical reasoning
- Evaluates practical AI understanding capabilities

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should follow "ANSWER: [ANSWER]" format
- Uses step-by-step reasoning prompting
- Simple accuracy metric for evaluation
- Tests models on practical, real-world scenarios
""",
        dataset_id='lmms-lab/RealWorldQA',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=OPEN_PROMPT,
    )
)
class RealWorldQAAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        content_list: list[Content] = [ContentText(text=OPEN_PROMPT.format(question=record['question']))]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='webp', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['answer'],
            metadata={'image_path': record['image_path']}
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        pattern = r'ANSWER:\s*(.*)'
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()
        return ''
