from __future__ import annotations

import base64
import binascii
import importlib
import requests
import time
from logging import getLogger
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ContentImage
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.argument_utils import get_secret_value
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import PIL_to_base64, bytes_to_base64
from evalscope.utils.model_utils import get_device
from evalscope.utils.uri_utils import data_uri_to_base64

logger = getLogger()


class Text2ImageAPI(ModelAPI):
    """Text-to-image provider for local pipelines and OpenAI-compatible services."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        self.base_url = (base_url or '').rstrip('/')
        self._use_service = bool(self.base_url)
        if self._use_service:
            self._init_service(api_key=api_key, model_args=model_args)
        else:
            self._init_local_model(model_args=model_args)

    def _init_local_model(self, model_args: Dict[str, Any]) -> None:
        check_import(
            ['torch', 'torchvision', 'diffusers'],
            package='evalscope[aigc]',
            raise_error=True,
            feature_name='text2image',
        )
        import torch

        model_path = model_args.pop('model_path', None)
        torch_dtype = model_args.pop('precision', None) or model_args.pop('torch_dtype', None)
        device_map = model_args.pop('device_map', None)
        # torch dtype
        DTYPE_MAP = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16, 'auto': 'auto'}

        if isinstance(torch_dtype, str) and torch_dtype != 'auto':
            torch_dtype = DTYPE_MAP.get(torch_dtype, torch.float32)
        self.torch_dtype = torch_dtype
        self.device = device_map or get_device()

        self.pipeline_cls = model_args.pop('pipeline_cls', None)
        # default to DiffusionPipeline if not specified
        if self.pipeline_cls is None:
            if 'flux' in self.model_name.lower():
                self.pipeline_cls = 'FluxPipeline'
            else:
                self.pipeline_cls = 'DiffusionPipeline'

        model_name_or_path = model_path or self.model_name

        # from modelscope import pipeline_cls
        module = getattr(importlib.import_module('modelscope'), self.pipeline_cls)
        logger.info(f'Loading model {model_name_or_path} with {self.pipeline_cls} ...')

        self.model = module.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            **model_args,
        )

        self.model.to(self.device)

    def _init_service(self, api_key: Optional[str], model_args: Dict[str, Any]) -> None:
        self.url = self._build_service_url(self.base_url)
        self.api_key = api_key
        self.timeout = float(model_args.pop('timeout', 120))
        extra_body = model_args.pop('extra_body', {})
        if extra_body is None:
            extra_body = {}
        if not isinstance(extra_body, dict):
            raise ValueError('model_args.extra_body must be a dictionary.')
        self._service_kwargs = {**model_args, **extra_body}
        self.session = requests.Session()

    @staticmethod
    def _build_service_url(base_url: str) -> str:
        endpoint = base_url.rstrip('/')
        if endpoint.endswith('/images/generations'):
            return endpoint
        return f'{endpoint}/images/generations'

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if self._use_service:
            return self._generate_service(input=input, config=config)

        # prepare generator
        kwargs: Dict[str, Any] = {}
        if config.height is not None:
            kwargs['height'] = config.height
        if config.width is not None:
            kwargs['width'] = config.width
        if config.num_inference_steps is not None:
            kwargs['num_inference_steps'] = config.num_inference_steps
        if config.guidance_scale is not None:
            kwargs['guidance_scale'] = config.guidance_scale
        # update with extra model parameters
        kwargs.update(config.model_extra)

        # assume the first text as prompt
        prompt = input[0].text
        # get the first image as output
        image = self.model(prompt=prompt, **kwargs).images[0]

        image_base64 = PIL_to_base64(image)

        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice.from_content(content=[ContentImage(image=image_base64)])],
            time=time.time(),
        )

    def _generate_service(self, input: List[ChatMessage], config: GenerateConfig) -> ModelOutput:
        start_time = time.monotonic()
        payload = self._service_payload(input=input, config=config)
        request_timeout = config.timeout if config.timeout is not None else self.timeout
        response = self.session.post(
            self.url,
            headers=self._service_headers(config.extra_headers),
            json=payload,
            params=get_secret_value(config.extra_query),
            timeout=request_timeout,
        )
        response.raise_for_status()

        response_payload = response.json()
        image_base64, metadata = self._parse_service_response(
            response_payload=response_payload,
            timeout=request_timeout,
        )
        request_id = response.headers.get('x-request-id')
        if request_id:
            metadata['request_id'] = request_id

        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice.from_content(content=[ContentImage(image=image_base64)])],
            time=time.monotonic() - start_time,
            metadata=metadata,
        )

    def _service_payload(self, input: List[ChatMessage], config: GenerateConfig) -> Dict[str, Any]:
        if not input or not input[0].text:
            raise ValueError('Text-to-image generation requires a non-empty prompt.')

        payload: Dict[str, Any] = {
            'model': self.model_name,
            'prompt': input[0].text,
        }
        if config.n is not None:
            payload['n'] = config.n
        if config.width is not None or config.height is not None:
            if config.width is None or config.height is None:
                raise ValueError('Both width and height are required when setting the image size.')
            payload['size'] = f'{config.width}x{config.height}'

        payload.update(self._service_kwargs)
        payload.update(config.extra_body or {})
        payload.update(config.model_extra or {})
        return payload

    def _service_headers(self, extra_headers: Optional[Dict[str, Any]]) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        raw_extra_headers = get_secret_value(extra_headers)
        if isinstance(raw_extra_headers, dict):
            headers.update({str(key): str(value) for key, value in raw_extra_headers.items()})
        return headers

    def _parse_service_response(self, response_payload: Any, timeout: float) -> tuple[str, Dict[str, Any]]:
        if not isinstance(response_payload, dict):
            raise ValueError('Image generation service returned a non-object JSON response.')

        data = response_payload.get('data')
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError('Image generation service response must contain a non-empty data list.')

        image = data[0]
        image_b64_json = image.get('b64_json')
        image_url = image.get('url')
        if isinstance(image_b64_json, str) and image_b64_json:
            image_base64 = data_uri_to_base64(image_b64_json)
            try:
                base64.b64decode(image_base64, validate=True)
            except (binascii.Error, ValueError, TypeError) as ex:
                raise ValueError('Image generation service returned invalid base64 image data.') from ex
        elif isinstance(image_url, str) and image_url:
            image_response = self.session.get(image_url, timeout=timeout)
            image_response.raise_for_status()
            image_base64 = bytes_to_base64(image_response.content)
        else:
            raise ValueError('Image generation service response must include b64_json or url.')

        metadata = {key: value for key, value in response_payload.items() if key != 'data'}
        if image.get('revised_prompt') is not None:
            metadata['revised_prompt'] = image['revised_prompt']
        return image_base64, metadata
