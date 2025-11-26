from __future__ import annotations

import copy
import functools
import json
import time
import torch  # type: ignore
from concurrent.futures import Future
from dataclasses import dataclass
from logging import getLogger
from modelscope import AutoModelForCausalLM, AutoTokenizer
from queue import Empty, Queue
from threading import Thread
from torch import Tensor  # type: ignore
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union, cast
from typing_extensions import override

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
from evalscope.utils.model_utils import get_device

logger = getLogger()


class ModelScopeAPI(ModelAPI):

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
        device_map = collect_model_arg('device_map')
        torch_dtype = collect_model_arg('precision')
        tokenizer_path = collect_model_arg('tokenizer_path')
        self.chat_template = collect_model_arg('chat_template')
        self.tokenizer_call_args = collect_model_arg('tokenizer_call_args')
        self.enable_thinking = collect_model_arg('enable_thinking')
        if self.tokenizer_call_args is None:
            self.tokenizer_call_args = {}

        # device
        self.device = device_map or get_device()

        # torch dtype
        DTYPE_MAP = {
            'float16': torch.float16,
            'torch.float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'torch.bfloat16': torch.bfloat16,
            'half': torch.half,
            'torch.half': torch.half,
            'float32': torch.float32,
            'torch.float32': torch.float32,
            'float64': torch.float64,
            'torch.float64': torch.float64,
            'auto': 'auto'
        }

        if isinstance(torch_dtype, str) and torch_dtype != 'auto':
            torch_dtype = DTYPE_MAP.get(torch_dtype, torch.float32)
        self.torch_dtype = torch_dtype

        # model
        model_name_or_path = model_path or model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=self.device,
            token=self.api_key,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            **model_args
        )

        # tokenizer
        tokenizer_name_or_path = tokenizer_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        # LLMs generally don't have a pad token and we need one for batching
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # add a pad token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # set padding side to left for LLMs
        self.tokenizer.padding_side = 'left'
        # set chat template if provided
        if self.chat_template:
            self.tokenizer.chat_template = self.chat_template
            logger.info(f'Using custom chat template: {self.chat_template}')

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:

        # create chat
        chat = self.ms_chat(input, tools)

        assert isinstance(self.tokenizer_call_args, dict)
        # prepare tokenizer
        tokenizer = functools.partial(
            self.tokenizer,
            return_tensors='pt',
            padding=True,
            **self.tokenizer_call_args,
        )

        # prepare generator
        kwargs: Dict[str, Any] = {}
        if config.do_sample is not None:
            kwargs['do_sample'] = config.do_sample
        if config.n is not None:
            if config.n > 1:
                assert config.do_sample, 'n > 1 requires do_sample=True in GenerateConfig'
            kwargs['num_return_sequences'] = config.n
        if config.max_tokens is not None:
            kwargs['max_new_tokens'] = config.max_tokens
        if config.temperature is not None:
            kwargs['temperature'] = config.temperature
        if config.top_p is not None:
            kwargs['top_p'] = config.top_p
        if config.top_k is not None:
            kwargs['top_k'] = config.top_k
        if config.logprobs is not None:
            kwargs['output_logits'] = config.logprobs
        if 'return_dict_in_generate' in kwargs:
            assert kwargs['return_dict_in_generate']
        if config.stop_seqs is not None:
            from transformers.generation import StopStringCriteria  # type: ignore

            stopping_criteria = [StopStringCriteria(self.tokenizer, config.stop_seqs)]
            kwargs['stopping_criteria'] = stopping_criteria

        # Handle extra_body parameters
        if config.extra_body:
            # Extract known parameters that should be passed to chat template
            self.enable_thinking = config.extra_body.get('enable_thinking', self.enable_thinking)

            # Pass through other extra_body parameters to generator if they're valid
            for key, value in config.extra_body.items():
                if key not in ['enable_thinking'] and key not in kwargs:
                    kwargs[key] = value

        kwargs['return_dict_in_generate'] = True
        generator = functools.partial(self.model.generate, **kwargs)

        # prepare decoder
        decoder = functools.partial(
            self.tokenizer.batch_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # generate
        responses = batched_generate(
            GenerateInput(
                input=chat,
                device=self.model.device,
                tokenizer=tokenizer,
                generator=generator,
                decoder=decoder,
                batch_size=config.batch_size or self.max_connections(),
            )
        )

        choices: List[ChatCompletionChoice] = []
        for response in responses:
            # gather logprobs
            final_logprobs = None
            if config.logprobs is not None:
                final_logprobs = extract_logprobs(
                    response=response,
                    top=config.top_logprobs,
                    tokenizer=self.tokenizer,
                )

            # construct choice
            # TODO: Handle tool calls
            choice = ChatCompletionChoice(
                message=ChatMessageAssistant(content=response.output, model=self.model_name, source='generate'),
                logprobs=(Logprobs(content=final_logprobs) if final_logprobs is not None else None),
                stop_reason=response.stop_reason,
            )
            choices.append(choice)

        # return output
        return ModelOutput(
            model=self.model_name,
            choices=choices,
            usage=ModelUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
            ),
            time=response.time,
        )

    @override
    def max_tokens(self) -> Optional[int]:
        """Default is 2048, bump it up to a value suitable for evals."""
        return 2048

    @override
    def max_connections(self) -> int:
        """Effectively the batch size."""
        return 8

    def ms_chat(self, messages: List[ChatMessage], tools: List[ToolInfo]) -> str:
        # convert to ms format
        tools_list = []
        ms_messages = copy.deepcopy(messages)
        if len(tools) > 0:
            tools_list = [json.loads(tool.model_dump_json(exclude_none=True, indent=2)) for tool in tools]

        ms_messages = message_content_to_string(ms_messages)
        # apply chat template
        if self.tokenizer.chat_template is not None:
            template_kwargs = {
                'add_generation_prompt': True,
                'tokenize': False,
            }
            if len(tools_list) > 0:
                template_kwargs['tools'] = tools_list
            if self.enable_thinking is not None:
                template_kwargs['enable_thinking'] = self.enable_thinking

            chat = self.tokenizer.apply_chat_template(
                ms_messages,
                **template_kwargs,
            )
        else:
            chat = ''
            for message in ms_messages:
                chat += f'{message.role}: {message.content}\n'
        # return
        return cast(str, chat)


def message_content_to_string(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Convert list of content in `ChatMessageAssistant`, `ChatMessageUser` or `ChatMessageSystem` to a string."""
    for message in messages:
        if isinstance(message.content, list):
            is_multimodal = any(
                isinstance(item, (ContentAudio, ContentImage, ContentVideo)) for item in message.content
            )
            if is_multimodal:
                raise NotImplementedError(
                    'Transformer model does not support multimodal content, please provide text inputs only.'
                )
            message.content = message.text
    return messages


# return value from generate as a result of specifying return_dict_in_generate
class ModelGenerateOutput:
    sequences: Tensor
    logits: tuple[Tensor]


class Tokenizer(Protocol):

    def __call__(self, input: List[str]) -> Dict[Literal['input_ids', 'attention_mask'], Tensor]:
        ...


class Generator(Protocol):

    def __call__(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        ...


class Decoder(Protocol):

    def __call__(self, sequences: Tensor) -> list[str]:
        ...


@dataclass
class GenerateInput:
    input: str
    device: str
    tokenizer: Tokenizer
    generator: Generator
    decoder: Decoder
    batch_size: int


@dataclass
class GenerateOutput:
    output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    logprobs: Optional[torch.Tensor]
    time: float
    stop_reason: Optional[str] = None


@dataclass
class _QueueItem:
    input: GenerateInput
    future: Future[GenerateOutput]


batch_thread: Optional[Thread] = None

batch_queue: 'Queue[_QueueItem]' = Queue()


def batched_generate(input: GenerateInput) -> List[GenerateOutput]:
    # start the background thread if necessary
    global batch_thread
    if batch_thread is None:
        batch_thread = Thread(target=process_batches, daemon=True)
        batch_thread.start()

    # enqueue the job
    future = Future[GenerateOutput]()
    batch_queue.put(_QueueItem(input=input, future=future))

    return future.result()


def process_batches() -> None:
    while True:
        # drain the queue (wait until no new messages have shown up for 2 seconds)
        inputs: List[Tuple[GenerateInput, Future[GenerateOutput]]] = []
        while True:
            try:
                input = batch_queue.get(timeout=2)
                inputs.append((input.input, input.future))
                if len(inputs) == input.input.batch_size:
                    # max batch size reached
                    break
            except Empty:
                # we have exhausted the queue
                break

        # see if we have any work to do
        if len(inputs) == 0:
            continue

        try:
            # capture the generator and decoder functions
            start_time = time.monotonic()
            first_input = inputs[0][0]
            device = first_input.device
            tokenizer = first_input.tokenizer
            generator = first_input.generator
            decoder = first_input.decoder
            num_return_sequences = generator.keywords.get('num_return_sequences', 1)
            max_new_tokens = generator.keywords.get('max_new_tokens', None)
            # In case some callers use max_length, honor it as a fallback:
            max_length = generator.keywords.get('max_length', None)

            # tokenize and move to device
            tokenized_inputs = tokenizer([item[0].input for item in inputs])
            input_ids = tokenized_inputs['input_ids']
            attention_mask = tokenized_inputs['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # generate
            with torch.inference_mode():
                generation_outputs = cast(
                    ModelGenerateOutput,
                    generator(input_ids=input_ids, attention_mask=attention_mask),
                )
                generate_ids = generation_outputs.sequences
                logits = generation_outputs.logits

            # get logprobs from logits
            logprobs = None
            if logits is not None:
                stacked_logits = torch.stack(logits).transpose(0, 1)
                logprobs = torch.nn.functional.log_softmax(stacked_logits, dim=-1)

            # decode
            generated_tokens = generate_ids[:, input_ids.size(dim=1):]
            if logprobs is not None:
                assert logprobs.shape[1] == generated_tokens.shape[1]
            outputs = decoder(sequences=generated_tokens)

            # call back futures
            total_time = time.monotonic() - start_time
            for input_index in range(len(inputs)):
                choices: List[GenerateOutput] = []
                # handle input
                future = inputs[input_index][1]
                input_tokens = input_ids[input_index].shape[-1]
                # handle choices
                for choice_index in range(num_return_sequences):
                    output_index = input_index * num_return_sequences + choice_index
                    # handle out of
                    output = outputs[output_index]
                    output_tokens = generate_ids[output_index].shape[-1] - input_tokens
                    logprobs_tensor = logprobs[output_index] if logprobs is not None else None

                    # Determine stop reason:
                    # 1) If the configured token limit was reached, treat as max_tokens.
                    #    - Prefer max_new_tokens; fallback to max_length when provided.
                    reached_max_tokens = False
                    if max_new_tokens is not None:
                        reached_max_tokens = output_tokens >= max_new_tokens
                    elif max_length is not None:
                        # max_length is total tokens (input + output)
                        reached_max_tokens = (input_tokens + output_tokens) >= max_length

                    if reached_max_tokens:
                        finish_reason = 'max_tokens'
                    else:
                        finish_reason = 'stop'  # covers EOS or stop string criteria

                    # create the output
                    choices.append(
                        GenerateOutput(
                            output=output,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                            logprobs=logprobs_tensor,
                            time=total_time,
                            stop_reason=finish_reason,
                        )
                    )

                # asyncio futures are not thread safe, so we need to pass the event loop
                # down to this point, so we can mark the future as done in a thread safe manner.
                # see: https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading
                future.set_result(choices)

        except Exception as ex:
            for inp in inputs:
                future = inp[1]
                future.set_exception(ex)


def extract_logprobs(
    response: GenerateOutput,
    top: Optional[int],
    tokenizer,
) -> List[Logprob]:
    assert response.logprobs is not None
    k = top or 1
    topk_values, topk_inds = response.logprobs.topk(k=k, dim=-1)
    final_logprobs = []
    for toks, vals in zip(topk_inds, topk_values):
        top_logprobs: List[TopLogprob] = []
        for tok, val in zip(toks, vals):
            # TODO: you get byte artifacts converting single ids to tokens like this...
            # but `tokenizer.decode` strips spaces. There must be a better way to do this.
            token_str = tokenizer.convert_ids_to_tokens(tok.item())
            top_logprobs.append(TopLogprob(
                token=token_str,
                logprob=val,
                bytes=list(map(ord, token_str)),
            ))
        final_logprobs.append(
            Logprob(
                token=top_logprobs[0].token,
                logprob=top_logprobs[0].logprob,
                bytes=top_logprobs[0].bytes,
                top_logprobs=top_logprobs,
            )
        )
    return final_logprobs
