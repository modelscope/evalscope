from __future__ import annotations

import inspect
import io
import os
import subprocess
import threading
import time
from logging import getLogger
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import numpy as np
from openai import OpenAI

from evalscope.api.messages import ChatMessage, ContentAudio
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.argument_utils import get_secret_value, get_supported_params
from evalscope.utils.function_utils import retry_call
from evalscope.utils.import_utils import check_import
from evalscope.utils.model_utils import get_device
from evalscope.utils.uri_utils import file_as_data, is_data_uri

from .openai_compatible import NON_RETRYABLE_OPENAI_ERRORS

logger = getLogger()

_FORMAT_TO_MIME: Dict[str, str] = {
    'flac': 'audio/flac',
    'm4a': 'audio/mp4',
    'mp3': 'audio/mpeg',
    'mp4': 'audio/mp4',
    'mpeg': 'audio/mpeg',
    'mpga': 'audio/mpeg',
    'ogg': 'audio/ogg',
    'opus': 'audio/ogg',
    'pcm': 'audio/pcm',
    'wav': 'audio/wav',
    'webm': 'audio/webm',
}

_MIME_TO_EXT: Dict[str, str] = {
    'audio/flac': 'flac',
    'audio/mp4': 'm4a',
    'audio/mpeg': 'mp3',
    'audio/ogg': 'ogg',
    'audio/pcm': 'pcm',
    'audio/wav': 'wav',
    'audio/webm': 'webm',
    'audio/x-flac': 'flac',
    'audio/x-m4a': 'm4a',
    'audio/x-wav': 'wav',
}

_ASR_TEXT_DELIMITER = '<asr_text>'


class Speech2TextAPI(ModelAPI):
    """Speech-to-text provider for OpenAI-compatible transcription services and local Transformers ASR checkpoints."""

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
        self._local_lock = threading.Lock()
        self._local_backend: Optional[str] = None

        if self._use_service:
            self._init_service(api_key=api_key, model_args=model_args)
        else:
            self._init_local_model(api_key=api_key, model_args=model_args)

    def _init_service(self, api_key: Optional[str], model_args: Dict[str, Any]) -> None:
        self.base_url = self._normalize_service_base_url(self.base_url)
        self.api_key = self._resolve_service_api_key(api_key)

        self.language: Optional[str] = model_args.pop('language', None)
        self.prompt: Optional[str] = model_args.pop('prompt', None)
        self.response_format: Optional[str] = model_args.pop('response_format', 'json')
        self.extra_body: Dict[str, Any] = dict(model_args.pop('extra_body', None) or {})

        client_params = get_supported_params(OpenAI) - {'api_key', 'base_url'}
        ignored_model_args = sorted(set(model_args) - client_params)
        if ignored_model_args:
            logger.warning(
                'Ignoring model_args unsupported by the OpenAI client in speech2text service mode: '
                f'{", ".join(ignored_model_args)}'
            )
        client_args = {key: value for key, value in model_args.items() if key in client_params}
        if client_args.get('max_retries') not in {None, 0}:
            logger.warning(
                'Ignoring model_args.max_retries in speech2text service mode; use generation_config.retries instead.'
            )
        client_args['max_retries'] = 0

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **client_args,
        )
        self._service_request_params = get_supported_params(self.client.audio.transcriptions.create)

    def _init_local_model(self, api_key: Optional[str], model_args: Dict[str, Any]) -> None:
        check_import(
            ['torch', 'transformers'],
            package=['torch', 'transformers'],
            raise_error=True,
            feature_name='speech2text',
        )
        import torch
        import transformers
        from transformers import AutoConfig

        model_path = model_args.pop('model_path', None)
        self.language = model_args.pop('language', None)
        self.prompt = model_args.pop('prompt', None)
        self.chunk_length_s = model_args.pop('chunk_length_s', None)
        self.generate_kwargs: Dict[str, Any] = dict(model_args.pop('generate_kwargs', None) or {})
        revision = model_args.pop('revision', None)
        trust_remote_code = bool(model_args.pop('trust_remote_code', False))
        attn_implementation = model_args.pop('attn_implementation', None)
        device_map = model_args.pop('device_map', None)
        torch_dtype = (
            model_args.pop('precision', None) or model_args.pop('torch_dtype', None) or model_args.pop('dtype', None)
        )
        token = self._resolve_local_token(api_key)

        self.torch_dtype = self._resolve_torch_dtype(torch_dtype, torch)
        self.device = device_map or get_device()
        model_name_or_path = model_path or self.model_name

        config_kwargs: Dict[str, Any] = {'trust_remote_code': trust_remote_code}
        if revision is not None:
            config_kwargs['revision'] = revision
        if token is not None:
            config_kwargs['token'] = token

        model_config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)

        if self._is_qwen3_asr_config(model_config):
            self._init_local_qwen3_asr(
                transformers=transformers,
                model_name_or_path=model_name_or_path,
                device_map=device_map,
                attn_implementation=attn_implementation,
                config_kwargs=config_kwargs,
                model_args=model_args,
            )
        else:
            self._init_local_pipeline(
                transformers=transformers,
                model_name_or_path=model_name_or_path,
                device_map=device_map,
                attn_implementation=attn_implementation,
                config_kwargs=config_kwargs,
                model_args=model_args,
            )

    def _init_local_qwen3_asr(
        self,
        transformers: Any,
        model_name_or_path: str,
        device_map: Optional[Any],
        attn_implementation: Optional[str],
        config_kwargs: Dict[str, Any],
        model_args: Dict[str, Any],
    ) -> None:
        model_cls = getattr(transformers, 'Qwen3ASRForConditionalGeneration', None)
        if model_cls is None:
            raise ImportError(
                f"Model '{model_name_or_path}' uses qwen3_asr (Qwen3ASRForConditionalGeneration), "
                "which requires transformers>=5.21. Please run `pip install -U 'transformers>=5.21'`."
            )

        from transformers import AutoProcessor

        self._local_backend = 'qwen3_asr'
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **config_kwargs)

        load_kwargs: Dict[str, Any] = {**config_kwargs, **model_args}
        if self.torch_dtype is not None:
            dtype_key = self._dtype_kwarg_name(model_cls.from_pretrained)
            load_kwargs[dtype_key] = self.torch_dtype
        if device_map is not None:
            load_kwargs['device_map'] = device_map
        if attn_implementation is not None:
            load_kwargs['attn_implementation'] = attn_implementation

        logger.info(f'Loading Qwen3-ASR model {model_name_or_path} ...')
        self.model = model_cls.from_pretrained(model_name_or_path, **load_kwargs)
        if device_map is None and hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def _init_local_pipeline(
        self,
        transformers: Any,
        model_name_or_path: str,
        device_map: Optional[Any],
        attn_implementation: Optional[str],
        config_kwargs: Dict[str, Any],
        model_args: Dict[str, Any],
    ) -> None:
        self._local_backend = 'pipeline'
        pipe_kwargs: Dict[str, Any] = {
            'model': model_name_or_path,
            'trust_remote_code': config_kwargs['trust_remote_code'],
            **model_args,
        }
        if 'revision' in config_kwargs:
            pipe_kwargs['revision'] = config_kwargs['revision']
        if 'token' in config_kwargs:
            pipe_kwargs['token'] = config_kwargs['token']
        if self.torch_dtype is not None:
            dtype_key = self._dtype_kwarg_name(transformers.pipeline)
            pipe_kwargs[dtype_key] = self.torch_dtype
        if device_map is not None:
            pipe_kwargs['device_map'] = device_map
        else:
            pipe_kwargs['device'] = self.device
        if attn_implementation is not None:
            model_kwargs = dict(pipe_kwargs.get('model_kwargs') or {})
            model_kwargs['attn_implementation'] = attn_implementation
            pipe_kwargs['model_kwargs'] = model_kwargs

        logger.info(f'Loading Transformers ASR pipeline for {model_name_or_path} ...')
        self.pipeline = transformers.pipeline('automatic-speech-recognition', **pipe_kwargs)

    @staticmethod
    def _normalize_service_base_url(base_url: str) -> str:
        base = base_url.rstrip('/')
        for suffix in ('/audio/transcriptions', '/chat/completions'):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        return base

    @staticmethod
    def _resolve_service_api_key(api_key: Optional[str]) -> str:
        if api_key and api_key != 'EMPTY':
            return api_key
        # EVALSCOPE_API_KEY first, matching Text2ImageAPI: an OPENAI_API_KEY meant for
        # api.openai.com must not be sent to a local or third-party service by default.
        return os.getenv('EVALSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY') or api_key or 'EMPTY'

    @staticmethod
    def _resolve_local_token(api_key: Optional[str]) -> Optional[str]:
        if api_key and api_key != 'EMPTY':
            return api_key
        return None

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: Any, torch_module: Any) -> Any:
        if torch_dtype is None:
            return None
        if isinstance(torch_dtype, str):
            dtype_map = {
                'auto': 'auto',
                'bfloat16': getattr(torch_module, 'bfloat16', None),
                'float16': getattr(torch_module, 'float16', None),
                'float32': getattr(torch_module, 'float32', None),
                'half': getattr(torch_module, 'float16', None),
                'torch.bfloat16': getattr(torch_module, 'bfloat16', None),
                'torch.float16': getattr(torch_module, 'float16', None),
                'torch.float32': getattr(torch_module, 'float32', None),
                'torch.half': getattr(torch_module, 'float16', None),
            }
            resolved = dtype_map.get(torch_dtype)
            return resolved if resolved is not None else torch_dtype
        return torch_dtype

    @staticmethod
    def _dtype_kwarg_name(callable_obj: Any) -> str:
        try:
            params = inspect.signature(callable_obj).parameters
        except (TypeError, ValueError):
            return 'torch_dtype'
        if 'dtype' in params and 'torch_dtype' not in params:
            return 'dtype'
        return 'torch_dtype'

    @staticmethod
    def _is_qwen3_asr_config(config: Any) -> bool:
        model_type = getattr(config, 'model_type', None)
        architectures = getattr(config, 'architectures', None) or []
        return model_type == 'qwen3_asr' or 'Qwen3ASRForConditionalGeneration' in architectures

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if self._use_service:
            return self._generate_service(input=input, config=config)
        return self._generate_local(input=input, config=config)

    def _generate_service(self, input: List[ChatMessage], config: GenerateConfig) -> ModelOutput:
        start_time = time.monotonic()
        filename, audio_bytes, mime_type = self._extract_audio(input)
        request = self._service_request(
            filename=filename,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            config=config,
        )
        response = retry_call(
            self.client.audio.transcriptions.create,
            retries=config.retries,
            sleep_interval=config.retry_interval or 0,
            no_retry_exceptions=NON_RETRYABLE_OPENAI_ERRORS,
            **request,
        )
        text, metadata = self._parse_service_response(response)
        return self._build_output(text=text, elapsed=time.monotonic() - start_time, metadata=metadata)

    def _service_request(
        self,
        filename: str,
        audio_bytes: bytes,
        mime_type: str,
        config: GenerateConfig,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            'model': self.model_name,
            'file': (filename, audio_bytes, mime_type),
        }
        if self.response_format is not None:
            request['response_format'] = self.response_format
        if self.language is not None:
            request['language'] = self.language
        if self.prompt is not None:
            request['prompt'] = self.prompt
        if config.temperature is not None:
            request['temperature'] = config.temperature

        request.update(config.model_extra or {})

        extra_body = dict(self.extra_body)
        if config.extra_body:
            extra_body.update(get_secret_value(config.extra_body))
        for key in list(request):
            if key not in self._service_request_params:
                extra_body[key] = request.pop(key)
        if extra_body:
            request['extra_body'] = extra_body
        if config.extra_query is not None:
            request['extra_query'] = get_secret_value(config.extra_query)
        if config.extra_headers is not None:
            request['extra_headers'] = get_secret_value(config.extra_headers)
        if config.timeout is not None:
            request['timeout'] = config.timeout
        return request

    @staticmethod
    def _parse_service_response(response: Any) -> tuple[str, Dict[str, Any]]:
        if isinstance(response, str):
            return response, {}

        if isinstance(response, dict):
            text = Speech2TextAPI._extract_text_from_payload(response)
            metadata = {key: value for key, value in response.items() if key != 'text'}
            return text, metadata

        metadata: Dict[str, Any] = {}
        payload: Optional[Dict[str, Any]] = None
        if hasattr(response, 'model_dump'):
            dumped = response.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                payload = dumped
                metadata = {key: value for key, value in dumped.items() if key != 'text'}

        text = getattr(response, 'text', None)
        if text is None and payload is not None:
            text = Speech2TextAPI._extract_text_from_payload(payload)
        if text is None:
            raise ValueError(f'No transcription text found in speech2text response: {response!r}')

        request_id = getattr(response, '_request_id', None)
        if request_id:
            metadata['request_id'] = request_id
        return str(text), metadata

    @staticmethod
    def _extract_text_from_payload(payload: Dict[str, Any]) -> str:
        text = payload.get('text')
        if text is None:
            text = payload.get('transcription')
        if text is None and isinstance(payload.get('result'), dict):
            text = payload['result'].get('text')
        if text is None:
            raise ValueError(f'No transcription text found in speech2text response: {payload}')
        return str(text)

    def _generate_local(self, input: List[ChatMessage], config: GenerateConfig) -> ModelOutput:
        start_time = time.monotonic()
        filename, audio_bytes, _ = self._extract_audio(input)
        audio_ext = os.path.splitext(filename)[1].lstrip('.').lower() or 'wav'

        if self._local_backend == 'qwen3_asr':
            text = self._generate_local_qwen3_asr(audio_bytes=audio_bytes, audio_ext=audio_ext, config=config)
        else:
            text = self._generate_local_pipeline(audio_bytes=audio_bytes, audio_ext=audio_ext, config=config)

        return self._build_output(
            text=text,
            elapsed=time.monotonic() - start_time,
            metadata={'backend': self._local_backend},
        )

    def _generate_local_qwen3_asr(self, audio_bytes: bytes, audio_ext: str, config: GenerateConfig) -> str:
        import torch

        feature_extractor = getattr(self.processor, 'feature_extractor', None)
        target_sr = int(getattr(feature_extractor, 'sampling_rate', 16000) or 16000)
        waveform, _ = self._decode_audio_waveform(audio_bytes, target_sr=target_sr, audio_ext=audio_ext)

        extra = config.model_extra or {}
        language = extra.get('language', self.language)
        prompt = extra.get('prompt', self.prompt)

        req_kwargs: Dict[str, Any] = {'audio': waveform, 'return_tensors': 'pt'}
        if language is not None:
            req_kwargs['language'] = language
        if prompt is not None:
            req_kwargs['prompt'] = prompt

        inputs = self.processor.apply_transcription_request(**req_kwargs)
        model_device = getattr(self.model, 'device', self.device)
        model_dtype = getattr(self.model, 'dtype', None)
        moved_inputs: Dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, 'to'):
                if hasattr(value, 'is_floating_point') and value.is_floating_point() and model_dtype is not None:
                    moved_inputs[key] = value.to(device=model_device, dtype=model_dtype)
                else:
                    moved_inputs[key] = value.to(device=model_device)
            else:
                moved_inputs[key] = value

        gen_kwargs = self._build_local_generate_kwargs(config)

        with self._local_lock, torch.inference_mode():
            generated = self.model.generate(**moved_inputs, **gen_kwargs)

        sequences = generated.sequences if hasattr(generated, 'sequences') else generated
        prompt_len = moved_inputs['input_ids'].shape[-1] if 'input_ids' in moved_inputs else 0
        new_tokens = sequences[:, prompt_len:]
        return self._decode_qwen3_asr_tokens(new_tokens[0])

    def _decode_qwen3_asr_tokens(self, token_ids: Any) -> str:
        try:
            text = self.processor.decode(token_ids, return_format='transcription_only')
        except TypeError:
            raw = self.processor.decode(token_ids, skip_special_tokens=True)
            text = self._strip_asr_text_prefix(str(raw))
        return str(text).strip()

    @staticmethod
    def _strip_asr_text_prefix(text: str) -> str:
        if _ASR_TEXT_DELIMITER in text:
            return text.split(_ASR_TEXT_DELIMITER, 1)[1]
        return text

    def _generate_local_pipeline(self, audio_bytes: bytes, audio_ext: str, config: GenerateConfig) -> str:
        waveform, sr = self._decode_audio_waveform(audio_bytes, target_sr=None, audio_ext=audio_ext)
        call_kwargs: Dict[str, Any] = {}
        if self.chunk_length_s is not None:
            call_kwargs['chunk_length_s'] = self.chunk_length_s

        gen_kwargs = self._build_local_generate_kwargs(config)
        extra = config.model_extra or {}
        language = extra.get('language', self.language)
        if language is not None and 'language' not in gen_kwargs:
            gen_kwargs['language'] = language
        if gen_kwargs:
            call_kwargs['generate_kwargs'] = gen_kwargs

        with self._local_lock:
            result = self.pipeline({'array': waveform, 'sampling_rate': sr}, **call_kwargs)

        if isinstance(result, dict):
            text = result.get('text')
            if text is None:
                raise ValueError(f'No transcription text found in local ASR pipeline result: {result}')
            return str(text)
        return str(result)

    def _build_local_generate_kwargs(self, config: GenerateConfig) -> Dict[str, Any]:
        gen_kwargs = dict(self.generate_kwargs)
        if config.max_tokens is not None:
            gen_kwargs['max_new_tokens'] = config.max_tokens
        if config.do_sample is not None:
            gen_kwargs['do_sample'] = config.do_sample
        if config.temperature is not None:
            gen_kwargs['temperature'] = config.temperature
        if config.top_p is not None:
            gen_kwargs['top_p'] = config.top_p
        if config.top_k is not None:
            gen_kwargs['top_k'] = config.top_k
        if config.extra_body:
            for key, value in get_secret_value(config.extra_body).items():
                gen_kwargs.setdefault(key, value)
        return gen_kwargs

    def _build_output(self, text: str, elapsed: float, metadata: Dict[str, Any]) -> ModelOutput:
        output = ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice.from_content(text)],
            time=elapsed,
            metadata=metadata,
        )
        output.message.perf_metrics = PerformanceMetrics(latency=elapsed, ttft=None)
        return output

    @staticmethod
    def _extract_audio(input: List[ChatMessage]) -> tuple[str, bytes, str]:
        audio_item: Optional[ContentAudio] = None
        for message in input:
            if isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, ContentAudio):
                        audio_item = part
                        break
            if audio_item is not None:
                break

        if audio_item is None:
            raise ValueError('Speech-to-text transcription requires a ContentAudio input.')

        declared_format = (audio_item.format or '').lower().lstrip('.')
        default_mime = _FORMAT_TO_MIME.get(declared_format, 'audio/wav')
        audio_bytes, mime_type = file_as_data(audio_item.audio, default_mime_type=default_mime)
        if not audio_bytes:
            raise ValueError('Speech-to-text audio input is empty.')

        ext = declared_format or _MIME_TO_EXT.get(mime_type.lower(), 'wav')
        if is_data_uri(audio_item.audio):
            filename = f'audio.{ext}'
        else:
            parsed_path = urlsplit(audio_item.audio).path
            basename = os.path.basename(parsed_path)
            if not basename:
                filename = f'audio.{ext}'
            elif '.' not in basename:
                filename = f'{basename}.{ext}'
            else:
                filename = basename

        return filename, audio_bytes, mime_type

    @staticmethod
    def _decode_audio_waveform(
        audio_bytes: bytes,
        target_sr: Optional[int] = 16000,
        audio_ext: str = 'wav',
    ) -> tuple[np.ndarray, int]:
        check_import('soundfile', package='soundfile', raise_error=True, feature_name='local speech2text')
        import soundfile as sf

        try:
            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=False)
        except Exception as sf_err:
            waveform, sr = Speech2TextAPI._decode_audio_via_ffmpeg(
                audio_bytes=audio_bytes,
                target_sr=target_sr or 16000,
                audio_ext=audio_ext,
                original_error=sf_err,
            )

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        if target_sr is not None and sr != target_sr and waveform.size > 0:
            waveform = Speech2TextAPI._resample_waveform(waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return np.ascontiguousarray(waveform, dtype=np.float32), int(sr)

    @staticmethod
    def _decode_audio_via_ffmpeg(
        audio_bytes: bytes,
        target_sr: int,
        audio_ext: str,
        original_error: Exception,
    ) -> tuple[np.ndarray, int]:
        cmd = [
            'ffmpeg',
            '-v',
            'error',
            '-i',
            'pipe:0',
            '-ac',
            '1',
            '-ar',
            str(target_sr),
            '-f',
            'f32le',
            'pipe:1',
        ]
        try:
            proc = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f'Failed to decode {audio_ext!r} audio bytes with soundfile ({original_error}) '
                f'and ffmpeg fallback failed: {exc}'
            ) from original_error
        return np.frombuffer(proc.stdout, dtype=np.float32), target_sr

    @staticmethod
    def _resample_waveform(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        try:
            import soxr

            return np.asarray(soxr.resample(waveform, orig_sr, target_sr), dtype=np.float32)
        except ImportError:
            pass
        new_length = max(1, int(round(len(waveform) * target_sr / orig_sr)))
        x_old = np.linspace(0.0, 1.0, num=len(waveform), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        return np.interp(x_new, x_old, waveform).astype(np.float32)
