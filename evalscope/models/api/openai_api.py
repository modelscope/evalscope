# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import time
import requests
from typing import Union, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
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

        with ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(self._generate, inputs))
        return results

    def _generate(self, messages: Union[str, List[Dict]]) -> str:

        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]

        max_num_retries = 0
        while max_num_retries < self.retry:
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
                resp = raw_response.json()

                if self.verbose:
                    logger.debug(f'>>raw_resp: {resp}')

                if self.logprobs:
                    return resp['choices']
                else:
                    if self.is_chat:
                        return resp['choices'][0]['message']['content'].strip()
                    else:
                        return resp['choices'][0]['text'].strip()

            except Exception as e:
                logger.error(e)
                max_num_retries += 1
                continue
            #
            # except requests.ConnectionError:
            #     logger.error('Got connection error, retrying...')
            #     max_num_retries += 1
            #     continue
            # try:
            #     response = raw_response.json()
            #
            # except requests.JSONDecodeError:
            #     logger.error('JsonDecode error, got', str(raw_response.content))
            #     max_num_retries += 1
            #     continue
            # logger.debug(str(response))
            # try:
            #     if self.logprobs:
            #         return response['choices']
            #     else:
            #         if self.is_chat:
            #             return response['choices'][0]['message']['content'].strip()
            #         else:
            #             return response['choices'][0]['text'].strip()
            # except KeyError:
            #     if 'error' in response:
            #         if response['error']['code'] == 'rate_limit_exceeded':
            #             time.sleep(10)
            #             logger.warning('Rate limit exceeded, retrying...')
            #             max_num_retries += 1
            #             continue
            #         elif response['error']['code'] == 'invalid_prompt':
            #             logger.error('Invalid inputs:', messages)
            #             return ''

        raise RuntimeError(f'Calling OpenAI failed after retrying for {max_num_retries} times.')
