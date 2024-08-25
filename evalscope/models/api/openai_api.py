# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import threading
import time
from asyncio import Queue

import requests
from typing import Union, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from modelscope.utils.logger import get_logger

logger = get_logger()


class OpenaiApi:

    def __init__(self,
                 model: str,
                 openai_api_key,
                 openai_api_base,
                 logprobs: Optional[bool] = False,
                 top_logprobs: Optional[int] = None,
                 max_new_tokens: int = 4096,
                 temperature: Optional[float] = 0.0,
                 is_chat: bool = True,
                 verbose: bool = False,
                 retry: int = 3,
                 query_per_second: int = 10,     # TODO
                 **kwargs):

        self.temperature = temperature
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

    def generate(self,
                 inputs: Union[List[str], List[List]],
                 **kwargs) -> List[str]:
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
            self.wait()

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
                    )

                def remove_none_val(input_d: dict):
                    return {k: v for k, v in input_d.items() if v is not None}
                data = remove_none_val(data)

                raw_response = requests.post(self.url,
                                             headers=header,
                                             data=json.dumps(data))

            except requests.ConnectionError:
                logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
                # print(f'>> raw_resp: {raw_response.json()}')
            except requests.JSONDecodeError:
                logger.error('JsonDecode error, got', str(raw_response.content))
                continue

            if self.verbose:
                logger.debug(str(response))
            try:
                if self.logprobs:
                    return response['choices']
                else:
                    if self.is_chat:
                        return response['choices'][0]['message']['content'].strip()
                    else:
                        return response['choices'][0]['text'].strip()
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(10)
                        logger.warn('Rate limit exceeded, retrying...')
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        logger.warning(f'insufficient_quota key: {self.openai_api_key}')
                        continue
                    elif response['error']['code'] == 'invalid_prompt':
                        logger.warning(f'Invalid prompt: {messages}')
                        return ''
                    elif response['error']['type'] == 'invalid_prompt':
                        logger.warning(f'Invalid prompt: {messages}')
                        return ''

                    logger.error('Find error message in response: ', str(response['error']))
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

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
