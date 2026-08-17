# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='general_vmcq',
        pretty_name='General-VMCQ',
        description="""
## Overview

General-VMCQ is a customizable visual multiple-choice question answering benchmark for multimodal models.
It uses MMMU-style format with image/video placeholders in text, supporting flexible media inputs.

## Task Description

- **Task Type**: Visual Multiple-Choice Question Answering
- **Input**: Question with `<image N>`/`<video N>` placeholders + choice options + media
- **Output**: Selected answer choice
- **Flexibility**: Supports custom datasets via local files

## Key Features

- MMMU-style format (not OpenAI message format)
- Supports up to 100 images and 100 videos per sample
- Flexible image/video input (path, URL, or base64 data URL)
- Chain-of-thought prompt template option
- Custom dataset support via local file loading

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- Train split: **dev**, Eval split: **val**
- Images/videos are plain strings (do not wrap in `{{"url": ...}}`)
- See [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html) for dataset format
""",  # noqa: E501
        tags=[Tags.MULTIPLE_CHOICE, Tags.CUSTOM, Tags.MULTI_MODAL],
        dataset_id='general_vmcq',
        subset_list=['default'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='dev',
        eval_split='val',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class GeneralVMCQAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    """
    General VMCQ (Visual Multiple Choice Question) Adapter for custom multimodal evaluation.

    Image data format example (JSONL/TSV):
    {
        "question": "<image 1> What animal is this?",
        "options": ["Dog", "Cat", "Tiger", "Elephant"],
        "image_1": "custom_eval/multimodal/images/dog.jpg",
        # or use a Hugging Face Image undecoded Sequence(Image()) 'images' column
        # "images": [{"path": "custom_eval/multimodal/images/dog.jpg"}, {"bytes": ...}]
        "answer": "A"
    }
    Video data format example:
    {
        "question": "<video 1> What type of media is provided?",
        "options": ["Image", "Audio", "Video", "Text"],
        "video_1": "custom_eval/multimodal/videos/sample.mp4",
        "answer": "C"
    }
    - JSONL/CSV/TSV inputs typically use plain image/video strings: base64 data
        URLs or local/remote paths.
    - Parquet or Hugging Face Image columns may also provide image dicts with
        ``{"bytes": ...}`` or ``{"path": ...}``, which are converted to data URLs.
    - 'options' is a list (JSON array) of strings; do NOT include "A.", "B." prefixes.
    - 'answer' is the correct letter (e.g., 'A').
    """  # noqa: E501

    MAX_IMAGES: int = 100
    MAX_VIDEOS: int = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        content_list, answers_list = self.create_content_and_answers_list(record)

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=record['answer'],
        )

    def _extract_images(self, record: Dict[str, Any]) -> Dict[int, str]:
        image_map: Dict[int, Any] = {}
        # prefer 'image_n' > 'images'
        if any(record.get(f'image_{i + 1}') for i in range(GeneralVMCQAdapter.MAX_IMAGES)):
            for i in range(GeneralVMCQAdapter.MAX_IMAGES):
                image_map[i + 1] = record.get(f'image_{i + 1}')
        elif record.get('images'):
            for i, image in enumerate(record['images']):
                image_map[i + 1] = image
        else:
            return {}

        for k, v in image_map.items():
            # Hugging Face Image columns may surface undecoded values as dicts.
            if isinstance(v, dict):
                if v.get('bytes') is not None:
                    bytes_obj = v['bytes']
                elif v.get('path'):
                    with open(v['path'], 'rb') as f:
                        bytes_obj = f.read()
                else:
                    raise ValueError(f"Image {k} must contain either 'path' or 'bytes', got {v}")

                # generally, guessing from bytes is more reliable than from path extension
                image_map[k] = self._image_bytes_to_base64(bytes_obj, guess_mimetype=True)
            elif v is not None and not isinstance(v, str):
                raise TypeError(f'Expect Image {k} as string (path, URL, or base64) or undecoded dict, got {type(v)}')
        return image_map

    def _extract_videos(self, record: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        video_map: Dict[int, Dict[str, Any]] = {}
        for i in range(GeneralVMCQAdapter.MAX_VIDEOS):
            video = record.get(f'video_{i + 1}')
            if video:
                video_map[i + 1] = {
                    'url': video,
                    'format': record.get(f'video_{i + 1}_format'),
                }
        return video_map

    def create_content_and_answers_list(self, record: Dict[str, Any]) -> tuple[List[Content], List[str]]:
        """
        Create a list of content elements and a list of answers from a record.
        Media are inserted at their placeholder positions in the text.

        Args:
            record (dict): The record containing question, media, and options.

        Returns:
            tuple: (content_list, answers_list)
        """
        # Prepare image map
        image_map = self._extract_images(record)
        video_map = self._extract_videos(record)

        raw_options = record.get('options')
        answers_list: List[str]
        if isinstance(raw_options, list):
            answers_list = [str(x) for x in raw_options]
        elif isinstance(raw_options, str):
            # Try JSON first, then fallback to Python literal list
            try:
                parsed = json.loads(raw_options)
                if not isinstance(parsed, list):
                    raise ValueError('options JSON is not a list')
                answers_list = [str(x) for x in parsed]
            except Exception:
                answers_list = [str(x) for x in ast.literal_eval(raw_options)]
        else:
            raise ValueError('Unsupported options format; expected list or JSON string of list')
        full_text = prompt(question=record['question'], choices=answers_list, template=self.prompt_template)
        content_list = self._parse_text_with_media(full_text, image_map=image_map, video_map=video_map)
        return content_list, answers_list
