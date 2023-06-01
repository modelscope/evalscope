# Copyright (c) Alibaba, Inc. and its affiliates.

import requests
from requests import Session
from requests.adapters import HTTPAdapter, Retry


def get_gpt_result():
    """
    GPT api calling.
    """

    session = Session()
    retry = Retry(
        total=2,
        read=2,
        connect=2,
        backoff_factor=1,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    api_url = f'https://api.mit-spider.alibaba-inc.com/chatgpt/api/ask'

    dataset_info = dict(
        query='hello, who are you ?',
    )
    data = dict(
        data=dataset_info,
    )

    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer {token}'}

    r = session.post(url=api_url, json=data, headers=headers)

    print('>>>resp: ', r)


if __name__ == '__main__':
    get_gpt_result()
