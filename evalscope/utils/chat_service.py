import os
import time
import torch
from contextlib import contextmanager
from functools import partial
from pydantic import BaseModel, Field
from threading import Thread
from typing import Any, List, Literal, Optional, Union


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelCard(BaseModel):
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'owner'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str
    reasoning_content: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[List[ChatMessage], str]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = 2048
    min_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length']


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal['chat.completion', 'chat.completion.chunk', 'images.generations']
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice, Any]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[Usage] = None


class TextCompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 2048
    min_tokens: Optional[int] = None


class TextCompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal['stop', 'length']


class TextCompletionResponse(BaseModel):
    model: str
    object: Literal['text_completion']
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    choices: List[TextCompletionResponseChoice]
    usage: Optional[Usage]


class ChatService:

    def __init__(self, model_path, attn_implementation):
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        from transformers import TextIteratorStreamer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype='auto',
            attn_implementation=attn_implementation,
        ).eval()
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        self.model_id = os.path.basename(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def count_tokens(self, text: str) -> int:
        # Use the tokenizer to count the number of tokens
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    async def list_models(self):
        model_card = ModelCard(id=self.model_id)
        return ModelList(data=[model_card])

    async def _chat(self, request: ChatCompletionRequest):
        formatted_prompt, inputs, prompt_tokens = self._prepare_chat_inputs(request)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            min_new_tokens=request.min_tokens,
            temperature=request.temperature,
        )
        outputs = outputs[0][prompt_tokens:]  # remove prompt
        completion_tokens = len(outputs)
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)

        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response),
            finish_reason='stop',
        )
        return ChatCompletionResponse(
            model=self.model_id,
            choices=[choice_data],
            object='chat.completion',
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _text_completion(self, request: TextCompletionRequest):
        inputs, prompt_tokens = self._prepare_text_inputs(request)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            min_new_tokens=request.min_tokens,
            temperature=request.temperature,
        )
        outputs = outputs[0][prompt_tokens:]  # remove prompt
        completion_tokens = len(outputs)
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)

        choice_data = TextCompletionResponseChoice(
            index=0,
            text=response,
            finish_reason='stop',
        )
        return TextCompletionResponse(
            model=self.model_id,
            choices=[choice_data],
            object='text_completion',
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def _prepare_text_inputs(self, request: TextCompletionRequest):
        inputs = self.tokenizer(request.prompt, return_tensors='pt', padding=False).to(self.device)
        prompt_tokens = len(inputs['input_ids'][0])
        return inputs, prompt_tokens

    def _stream_chat(self, request: ChatCompletionRequest):
        formatted_prompt, inputs, prompt_tokens = self._prepare_chat_inputs(request)
        completion_tokens = 0

        yield self._create_initial_chunk()

        generation_kwargs = dict(
            **inputs,
            streamer=self.streamer,
            max_new_tokens=request.max_tokens,
            min_new_tokens=request.min_tokens,
            temperature=request.temperature,
        )
        generate_partial = partial(self.model.generate, **generation_kwargs)

        with self._start_generation_thread(generate_partial):
            for new_text in self.streamer:
                yield self._create_chunk(new_text)
                completion_tokens += self.count_tokens(new_text)

        yield self._create_final_chunk(prompt_tokens, completion_tokens)
        yield '[DONE]'

    def _prepare_chat_inputs(self, request: ChatCompletionRequest):
        formatted_prompt = self.tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt', padding=False).to(self.device)
        prompt_tokens = len(inputs['input_ids'][0])
        return formatted_prompt, inputs, prompt_tokens

    @contextmanager
    def _start_generation_thread(self, generate_partial):
        thread = Thread(target=generate_partial)
        thread.start()
        try:
            yield
        finally:
            thread.join()

    def _create_initial_chunk(self):
        choice_data = ChatCompletionResponseStreamChoice(index=0, delta={'role': 'assistant'}, finish_reason=None)
        chunk = ChatCompletionResponse(
            model=self.model_id,
            choices=[choice_data],
            object='chat.completion.chunk',
            usage=None,
        )
        return chunk.model_dump_json(exclude_unset=True)

    def _create_chunk(self, new_text):
        choice_data = ChatCompletionResponseStreamChoice(index=0, delta={'content': new_text}, finish_reason=None)
        chunk = ChatCompletionResponse(
            model=self.model_id,
            choices=[choice_data],
            object='chat.completion.chunk',
            usage=None,
        )
        return chunk.model_dump_json(exclude_unset=True)

    def _create_final_chunk(self, prompt_tokens, completion_tokens):
        choice_data = ChatCompletionResponseStreamChoice(index=0, delta={}, finish_reason='stop')
        chunk = ChatCompletionResponse(
            model=self.model_id,
            choices=[choice_data],
            object='chat.completion.chunk',
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return chunk.model_dump_json(exclude_unset=True)
