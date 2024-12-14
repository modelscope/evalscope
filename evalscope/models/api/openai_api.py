# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import requests
import threading
import time
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from evalscope.utils.logger import get_logger

logger = get_logger()


class OpenaiApi:

    def __init__(
            self,
            model: str,
            openai_api_key,
            openai_api_base,
            logprobs: Optional[bool] = False,
            top_logprobs: Optional[int] = None,
            max_new_tokens: int = 4096,
            temperature: Optional[float] = 0.0,
            repetition_penalty: Optional[float] = 1.0,
            is_chat: bool = True,
            verbose: bool = True,
            retry: int = 3,
            query_per_second: int = 10,  # TODO
            **kwargs):

        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_new_tokens
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

        self.openai_api_key = openai_api_key
        self.url = openai_api_base
        self.model = model
        self.is_chat = is_chat
        self.retry = retry
        self.verbose = verbose

        self.token_bucket = TokenBucket(query_per_second, verbose)

    def generate_simple(self, inputs: Union[List[str]], num_proc: int = 8):

        def process_one(in_data: str):

            if self.is_chat:
                data = dict(
                    model=self.model,
                    messages=[{
                        'role': 'user',
                        'content': in_data
                    }],
                    max_tokens=self.max_tokens,
                    n=1,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    stop=None,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                )
            else:
                data = dict(
                    model=self.model,
                    prompt=in_data,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                )

            # todo
            openai_api_key = self.openai_api_key or ''
            header = {
                'Authorization': f'Bearer {openai_api_key}',
                'content-type': 'application/json',
            }
            data = json.dumps(data, ensure_ascii=False)

            if self.verbose:
                logger.info(f'>>data in generate_simple: {data}')

            resp = requests.post(self.url, headers=header, data=data)
            resp = resp.json()
            if self.verbose:
                logger.info(f'>>resp in generate_simple: {resp}')

            if self.logprobs:
                return resp['choices']
            else:
                if self.is_chat:
                    return resp['choices'][0]['message']['content'].strip()
                else:
                    return resp['choices'][0]['text'].strip()

        results = []
        with ThreadPoolExecutor(max_workers=num_proc) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_one, input_one): input_one for input_one in inputs}

            # Show progress bar
            for future in tqdm(as_completed(future_to_task), total=len(inputs)):
                results.append(future.result())

        return results

    def generate(self, inputs: Union[List[str], List[List]], **kwargs) -> List[str]:
        """
        Generate responses from OpenAI API.

        Args:
            inputs: The input messages for the model. It can be a string or a list of messages.
                e.g. ['who are you ?', 'what is your name ?']
                e.g. [[{'role': 'user', 'content': 'who are you ?'}], ...]
            kwargs: The optional arguments for the model.
        """
        results = []
        # with ThreadPoolExecutor() as executor:
        #     results = list(executor.map(self._generate, inputs))

        for input in inputs:
            results.append(self._generate(input))

        return results

    def _generate(self, messages: Union[str, List[Dict]]) -> str:

        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]

        max_num_retries = 0
        while max_num_retries < self.retry:
            # self.wait()

            header = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'content-type': 'application/json',
            }

            try:
                if self.is_chat:
                    data = dict(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        n=1,
                        logprobs=self.logprobs,
                        top_logprobs=self.top_logprobs,
                        stop=None,
                        temperature=self.temperature,
                        repetition_penalty=self.repetition_penalty,
                    )
                else:
                    # TODO: This is a temporary solution for non-chat models.
                    input_prompts = []
                    for msg in messages:
                        input_prompts.append(msg['content'])

                    data = dict(
                        model=self.model,
                        prompt='\n'.join(input_prompts),
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        repetition_penalty=self.repetition_penalty,
                    )

                def remove_none_val(input_d: dict):
                    return {k: v for k, v in input_d.items() if v is not None}

                data = remove_none_val(data)

                if self.verbose:
                    logger.info(f'>> Post data: {json.dumps(data, ensure_ascii=False)}')
                raw_response = requests.post(self.url, headers=header, data=json.dumps(data, ensure_ascii=False))

                response = raw_response.json()
                if self.verbose:
                    logger.info(f'>> response: {response}')

                if self.logprobs:
                    return response['choices']
                else:
                    if self.is_chat:
                        return response['choices'][0]['message']['content'].strip()
                    else:
                        return response['choices'][0]['text'].strip()

            except Exception as e:
                logger.error(f'Error occurs: {str(e)}')
                max_num_retries += 1
                continue

    def wait(self):
        return self.token_bucket.get_token()


class TokenBucket:
    """A token bucket for rate limiting.

    Args:
        query_per_second (float): The rate of the token bucket.
    """

    def __init__(self, rate, verbose=False):
        self._rate = rate
        self._tokens = threading.Semaphore(0)
        self.started = False
        self._request_queue = Queue()
        self.logger = get_logger()
        self.verbose = verbose

    def _add_tokens(self):
        """Add tokens to the bucket."""
        while True:
            if self._tokens._value < self._rate:
                self._tokens.release()
            time.sleep(1 / self._rate)

    def get_token(self):
        """Get a token from the bucket."""
        if not self.started:
            self.started = True
            threading.Thread(target=self._add_tokens, daemon=True).start()
        self._tokens.acquire()
        if self.verbose:
            cur_time = time.time()
            while not self._request_queue.empty():
                if cur_time - self._request_queue.queue[0] > 60:
                    self._request_queue.get()
                else:
                    break
            self._request_queue.put(cur_time)
            self.logger.info(f'Current RPM {self._request_queue.qsize()}.')
