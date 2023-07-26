import sys
import warnings
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerationConfig

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache


class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True
    model = None
    tokenizer = None

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get('revision', 'main')
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, revision=revision, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def load_compress_model(self, model_path, from_pretrained_kwargs: dict):
        model, tokenizer = self.load_model(model_path, from_pretrained_kwargs)
        model = model.quantize(8)
        return model, tokenizer

    def chat(self, prompt: str, history: List[Tuple[str, str]] = [], **kwargs):
        """Chat with the model."""
        return self.model.chat(self.tokenizer, prompt, history, **kwargs)


model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter
    raise ValueError(f'No valid model adapter for {model_path}')


def load_model(
    model_path: str,
    device: str = 'cuda',
    num_gpus: int = 1,
    load_8bit: bool = False,
    revision: str = 'main',
    debug: bool = False,
):
    """Load a model from Hugging Face."""

    # get model adapter
    adapter = get_model_adapter(model_path)

    kwargs = {'revision': revision}
    if device == 'cuda':
        kwargs['torch_dtype'] = torch.float16
        if num_gpus != 1:
            kwargs['device_map'] = 'auto'
    else:
        raise ValueError(f'Invalid device: {device}')

    if load_8bit and num_gpus != 1:
        warnings.warn(
            '8-bit quantization is not supported for multi-gpu inference.')

    # Load model
    model, tokenizer = (
        adapter.load_compress_model(model_path, kwargs) if load_8bit
        and num_gpus == 1 else adapter.load_model(model_path, kwargs))

    if device == 'cuda' and num_gpus == 1:
        model.to(device)

    if debug:
        print(model)

    adapter.model = model
    adapter.tokenizer = tokenizer
    return adapter


class ChatGLMAdapter(BaseModelAdapter):
    """The model adapter for THUDM/chatglm-6b, THUDM/chatglm2-6b"""

    def match(self, model_path: str):
        return 'chatglm' in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get('revision', 'main')
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision)
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, **from_pretrained_kwargs)
        return model, tokenizer


class BaichuanAdapter(BaseModelAdapter):
    """The model adapter for baichuan-inc/baichuan-7B, baichuan-inc/baichuan-13B-chat"""

    def match(self, model_path: str):
        return 'baichuan' in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get('revision', 'main')
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        return model, tokenizer

    def _build_chat_input(self,
                          tokenizer,
                          messages: List[dict],
                          max_new_tokens: int = 0):
        max_new_tokens = max_new_tokens or self.model.generation_config.max_new_tokens
        max_input_tokens = self.model.config.model_max_length - max_new_tokens
        max_input_tokens = max(self.model.config.model_max_length // 2,
                               max_input_tokens)
        total_input, round_input = [], []
        for i, message in enumerate(messages[::-1]):
            content_tokens = tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = ([self.model.generation_config.user_token_id]
                               + content_tokens + round_input)
                if total_input and \
                        (len(total_input) + len(round_input) > max_input_tokens):
                    break
                else:
                    total_input = round_input + total_input
                    if len(total_input) >= max_input_tokens:
                        break
                    else:
                        round_input = []
            elif message['role'] == 'assistant':
                round_input = (
                    [self.model.generation_config.assistant_token_id]
                    + content_tokens
                    + [self.model.generation_config.eos_token_id]
                    + round_input)
            else:
                raise ValueError(
                    f"message role not supported yet: {message['role']}")
        total_input = total_input[-max_input_tokens:]  # truncate left
        total_input.append(self.model.generation_config.assistant_token_id)
        total_input = torch.LongTensor([total_input]).to(self.model.device)
        return total_input

    @torch.no_grad()
    def _chat(
        self,
        tokenizer,
        messages: List[dict],
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        **kwargs,
    ):
        input_ids = self._build_chat_input(tokenizer, messages, max_new_tokens)
        outputs = self.model.generate(
            input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        response = tokenizer.decode(
            outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response

    def chat(self, prompt: str, history: List[Tuple[str, str]] = [], **kwargs):
        messages = []
        for item in history:
            messages.append({'role': 'user', 'content': item[0]})
            messages.append({'role': 'assistant', 'content': item[1]})
        messages.append({'role': 'user', 'content': prompt})

        response = self._chat(self.tokenizer, messages, **kwargs)
        history = history + [(prompt, response)]
        return response, history


class InternLMChatAdapter(BaseModelAdapter):
    """The model adapter for internlm/internlm-chat-7b"""

    def match(self, model_path: str):
        return 'internlm-chat' in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get('revision', 'main')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        model = model.eval()
        if '8k' in model_path.lower():
            model.config.max_sequence_length = 8192
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision)
        return model, tokenizer


register_model_adapter(ChatGLMAdapter)
register_model_adapter(BaichuanAdapter)
register_model_adapter(InternLMChatAdapter)
