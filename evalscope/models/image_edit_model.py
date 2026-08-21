from __future__ import annotations

import importlib
import time
import torch
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union, cast

from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ContentAudio,
    ContentImage,
    ContentText,
    ContentVideo,
)
from evalscope.api.model import (
    ChatCompletionChoice,
    GenerateConfig,
    Logprob,
    Logprobs,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    TopLogprob,
)
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.io_utils import PIL_to_base64, base64_to_PIL
from evalscope.utils.model_utils import get_device

logger = getLogger()


class ImageEditAPI(ModelAPI):

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        # collect known model_args (then delete them so we can pass the rest on)
        def collect_model_arg(name: str) -> Optional[Any]:
            nonlocal model_args
            value = model_args.get(name, None)
            if value is not None:
                model_args.pop(name)
            return value

        model_path = collect_model_arg('model_path')
        torch_dtype = collect_model_arg('precision') or collect_model_arg('torch_dtype')
        device_map = collect_model_arg('device_map')
        # torch dtype
        DTYPE_MAP = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16, 'auto': 'auto'}

        if isinstance(torch_dtype, str) and torch_dtype != 'auto':
            torch_dtype = DTYPE_MAP.get(torch_dtype, torch.float32)
        self.torch_dtype = torch_dtype
        self.device = device_map or get_device()

        self.pipeline_cls = collect_model_arg('pipeline_cls')
        # default to DiffusionPipeline if not specified
        if self.pipeline_cls is None:
            if 'qwen' in model_name.lower():
                self.pipeline_cls = 'QwenImageEditPipeline'
            else:
                logger.error('Pipeline class not found. Please provide a valid `pipeline_cls` in model args.')
                raise ValueError('Invalid pipeline class.')

        model_name_or_path = model_path or model_name

        # from modelscope import pipeline_cls
        module = getattr(importlib.import_module('modelscope'), self.pipeline_cls)
        logger.info(f'Loading model {model_name_or_path} with {self.pipeline_cls} ...')

        self.model = module.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            **model_args,
        )

        self.model.to(self.device)

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:

        # prepare generator
        kwargs: Dict[str, Any] = {}
        if config.num_inference_steps is not None:
            kwargs['num_inference_steps'] = config.num_inference_steps
        kwargs.update(config.model_extra)

        # assume the first text as prompt
        content = input[0].content
        assert isinstance(content[0], ContentText) and isinstance(content[1], ContentImage), \
            'Invalid content types, expected (ContentText, ContentImage)'

        prompt = content[0].text
        input_image_base64 = content[1].image
        input_image = base64_to_PIL(input_image_base64)
        # get the first image as output
        output = self.model(image=input_image, prompt=prompt, **kwargs)
        image = output.images[0]

        image_base64 = PIL_to_base64(image)

        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice.from_content(content=[ContentImage(image=image_base64)])],
            time=time.time(),
        )
