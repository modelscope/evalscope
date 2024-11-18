import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from threading import Thread
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from modelscope import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field
from transformers import TextIteratorStreamer


@dataclass
class ServerSentEvent(object):

    def __init__(self, data='', event=None, id=None, retry=None):
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry

    @classmethod
    def decode(cls, line):
        """Decode line to ServerSentEvent


        Args:
            line (str): The line.

        Return:
            ServerSentEvent (obj:`ServerSentEvent`): The ServerSentEvent object.

        """
        if not line:
            return None
        sse_msg = cls()
        # format data:xxx
        field_type, _, field_value = line.partition(':')
        if field_value.startswith(' '):  # compatible with openai api
            field_value = field_value[1:]
        if field_type == 'event':
            sse_msg.event = field_value
        elif field_type == 'data':
            field_value = field_value.rstrip()
            sse_msg.data = field_value
        elif field_type == 'id':
            sse_msg.id = field_value
        elif field_type == 'retry':
            sse_msg.retry = field_value
        else:
            pass

        return sse_msg


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


class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = 2048
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
    object: Literal['chat.completion', 'chat.completion.chunk']
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[Usage]


class ChatService:

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map='auto', torch_dtype='auto')
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        self.model_id = os.path.basename(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        # Use the tokenizer to count the number of tokens
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    async def list_models(self):
        model_card = ModelCard(id=self.model_id)
        return ModelList(data=[model_card])

    async def _non_stream_predict(self, request: ChatCompletionRequest):
        formatted_prompt = self.tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt', padding=True).to(self.device)
        prompt_tokens = len(inputs['input_ids'][0])

        outputs = self.model.generate(
            **inputs,
            max_length=request.max_length,
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

    def _stream_predict(self, request: ChatCompletionRequest):
        formatted_prompt = self.tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt', padding=True).to(self.device)
        prompt_tokens = len(inputs['input_ids'][0])
        completion_tokens = 0

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(role='assistant'), finish_reason=None)
        chunk = ChatCompletionResponse(
            model=self.model_id,
            choices=[choice_data],
            object='chat.completion.chunk',
            usage=None,
        )
        yield f'data: {chunk.model_dump_json(exclude_unset=True)}\n\n'

        generation_kwargs = dict(
            **inputs,
            streamer=self.streamer,
            max_length=request.max_length,
            temperature=request.temperature,
        )
        generate_partial = partial(self.model.generate, **generation_kwargs)
        thread = Thread(target=generate_partial)
        thread.start()  # now start the thread

        for new_text in self.streamer:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
            completion_tokens += self.count_tokens(new_text)

            chunk = ChatCompletionResponse(
                model=self.model_id,
                choices=[choice_data],
                object='chat.completion.chunk',
                usage=None,
            )
            yield f'data: {chunk.model_dump_json(exclude_unset=True)}\n\n'

        choice_data = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(), finish_reason='stop')
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
        yield f'data: {chunk.model_dump_json(exclude_unset=True)}\n\n'

        thread.join()
        yield 'data: [DONE]\n\n'


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def create_app(args) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    chat_service = ChatService(model_path=args.model)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.get('/v1/models', response_model=ModelList)
    async def list_models():
        return await chat_service.list_models()

    @app.post('/v1/chat/completions')
    async def create_chat_completion(request: ChatCompletionRequest):
        if request.stream:
            return StreamingResponse(chat_service._stream_predict(request), media_type='text/event-stream')
        else:
            return await chat_service._non_stream_predict(request)

    return app


def start_app(args):
    app = create_app(args)
    uvicorn.run(app, host='0.0.0.0', port=8877, workers=1)


if __name__ == '__main__':
    from collections import namedtuple

    args = namedtuple('Args', 'model')

    start_app(args(model='Qwen/Qwen2.5-0.5B-Instruct'))
