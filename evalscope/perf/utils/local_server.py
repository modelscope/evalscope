import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from modelscope import AutoModel, AutoTokenizer
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse, ServerSentEvent


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    """Asynchronous context manager lifespan, used to clear CUDA cache"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache to release GPU memory
        torch.cuda.ipc_collect()


# By passing the lifespan context manager to the FastAPI constructor, you can manage GPU resources
# during the lifecycle of the FastAPI application and ensure necessary cleanup after each request is processed
app = FastAPI(lifespan=lifespan)

app.add_middleware(  # Use the add_middleware method to add middleware to the FastAPI application
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Parameter description:
# allow_origins=["*"]: List of allowed origins (domains). In this example, it is set to ["*"],
#   which means requests from any origin are allowed.
# allow_credentials=True: Indicates whether requests with credentials (such as cookies) are allowed.
# allow_methods=["*"]: List of allowed HTTP methods. In this example, it is set to ["*"],
#   which means all HTTP methods are allowed.
# allow_headers=["*"]: List of allowed request headers. In this example, it is set to ["*"],
#   which means all request headers are allowed.


class ModelCard(BaseModel):
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'owner'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


# Define the ModelList data model
class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = []


# Define the ChatMessage data model
class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str


# Define the DeltaMessage data model --> to store streaming output dialogue
class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


# Define the ChatCompletionRequest data model --> to store ChatCompletion's Request
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    # When stream is set to True in the ChatGPT API request, the API will generate chat responses in a streaming manner.
    stream: Optional[bool] = False


# Define the ChatCompletionResponseChoice data model --> to store ChatCompletionResponse's Choice
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length']


# Define the ChatCompletionResponseStreamChoice data model --> to store ChatCompletionResponseStream's Choice
class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


# Define the ChatCompletionResponse data model --> to store ChatCompletionResponse types
class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal['chat.completion', 'chat.completion.chunk']
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


# This code defines a route handler function list_models() to handle HTTP GET requests to the path "/v1/models".
# In this function, a ModelCard object is created and added to a ModelList object to be returned as the response.
@app.get('/v1/models', response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id='gpt-3.5-turbo')
    return ModelList(data=[model_card])


# Path: "/v1/chat/completions", Response model: ChatCompletionResponse
@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    # Limit questions to be asked only by the user role
    if request.messages[-1].role != 'user':
        raise HTTPException(status_code=400, detail='Invalid request')
    # Get the content of messages
    query = request.messages[-1].content

    # Get previous messages and check if it is a system message,
    # then combine the system message with the latest message
    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == 'system':
        # prev_messages.pop(0) removes the element at index 0 from the prev_messages list and returns its value.
        # As the element is removed from the list, the length of prev_messages is reduced by one element.
        query = prev_messages.pop(0).content + query

    # Pack each "user" question and "assistant" answer and store them in history (since prev_messages.pop(0),
    # there is no system message here)
    history = []
    if len(prev_messages) % 2 == 0:  # Check if there are other messages in prev_messages besides the system message
        for i in range(0, len(prev_messages), 2):
            if (prev_messages[i].role == 'user' and prev_messages[i + 1].role == 'assistant'):
                history.append([prev_messages[i].content, prev_messages[i + 1].content])

    if request.stream:  # If stream output is True
        generate = predict(query, history, request.model)  # Return streaming output result
        return EventSourceResponse(
            generate,
            media_type='text/event-stream')  # Use the EventSourceResponse class to construct the response object.

    # If not streaming, output the entire response directly,
    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=response),
        finish_reason='stop',
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object='chat.completion')


async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    # Define stream output settings
    choice_data = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(role='assistant'), finish_reason=None)
    # Store the stream output in the ChatCompletionResponse data model
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object='chat.completion.chunk')
    yield '{}'.format(chunk.model_dump_json(exclude_unset=True, ensure_ascii=False))  # Use yield for streaming output

    current_length = 0
    # Check the length of the streaming output text
    for new_response, _ in model.stream_chat(tokenizer, query, history):
        # If the length of the new response is equal to the current length,
        # it means no new text is generated, continue to the next loop
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]  # Get the newly added text part
        current_length = len(new_response)
        # Store the streaming output content in ChatCompletionResponseStreamChoice
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
        # Return the output content in real-time
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object='chat.completion.chunk')
        yield '{}'.format(chunk.model_dump_json(exclude_unset=True, ensure_ascii=False))

    # Return '[DONE]' after all output is complete
    choice_data = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(), finish_reason='stop')
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object='chat.completion.chunk')
    yield '{}'.format(chunk.model_dump_json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
    model = AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True).cuda()
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
