# Copyright (c) Alibaba, Inc. and its affiliates.
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import json
import os
import requests
import time
from dataclasses import dataclass, field
from rouge import Rouge
from urllib3.exceptions import MaxRetryError, NewConnectionError


def evaluate_rouge_l(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
    rougel = rouge_score['rouge-l']['f']
    return rougel


def nested_load_test_data(data_path):
    test_raw_data = []
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            temp_test = nested_load_test_data(os.path.join(data_path, f))
            test_raw_data += temp_test
        return test_raw_data
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        print('Load data from', data_path)
        temp_data = json.load(open(data_path, 'r'))
        test_raw_data = temp_data
        return test_raw_data
    else:
        return []


def baichuan_call(context: list, system: str):
    url = 'https://api.baichuan-ai.com/v1/chat/completions'
    api_key = 'sk-xxx'

    new_msg = []
    new_msg.append({'role': 'system', 'content': system})
    for m in context:
        if m['role'] == 'user':
            new_msg.append({'role': 'user', 'content': m['content']})
        elif m['role'] == 'function':
            new_msg.append({'role': 'user', 'content': m['content']})
        elif m['role'] == 'assistant':
            new_msg.append({'role': 'assistant', 'content': m['content']})
    # print(json.dumps(new_msg, indent=2))
    data = {'model': 'Baichuan2-Turbo', 'messages': new_msg, 'stream': False}

    json_data = json.dumps(data)

    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}

    for i in range(5):
        res = None
        try:
            res = requests.post(url, data=json_data, headers=headers, timeout=60)
            res = res._content.decode('utf-8')
            res = json.loads(res)
            return res['choices'][0]['message']['content']
        except KeyError:
            print(res)
            time.sleep(1)
            continue
        except ConnectionError:
            time.sleep(5)
            continue
        except MaxRetryError:
            time.sleep(5)
            continue
        except NewConnectionError:
            time.sleep(5)
            continue
    return ''


def minimax_call(context: list, system: str):
    group_id = 'your-id'
    api_key = 'your-xxx'

    url = f'https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId={group_id}'
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

    # construct message
    system_prompt = 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。' \
                    'MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'
    system_prompt += ('\n' + system)

    new_msg = []
    for m in context:
        if m['role'] == 'user':
            new_msg.append({'sender_type': 'USER', 'sender_name': 'user', 'text': m['content']})
        elif m['role'] == 'function':
            new_msg.append({'sender_type': 'USER', 'sender_name': 'funtion', 'text': m['content']})
        elif m['role'] == 'assistant':
            new_msg.append({'sender_type': 'BOT', 'sender_name': 'MM智能助理', 'text': m['content']})

    request_body = {
        'model': 'abab6-chat',
        # "model": "abab5.5s-chat",
        'tokens_to_generate': 8192,
        'reply_constraints': {
            'sender_type': 'BOT',
            'sender_name': 'MM智能助理'
        },
        'messages': new_msg,
        'bot_setting': [{
            'bot_name': 'MM智能助理',
            'content': system_prompt,
        }],
    }
    response = requests.post(url, headers=headers, json=request_body)
    status_code = response.status_code
    for i in range(5):
        try:
            if status_code == 200:
                reply = response.json()['reply']
                if len(reply) == 0:
                    print('limit rate')
                    time.sleep(8)
                    continue
                print(f'>>return: {reply}')
                return reply
            else:
                print(response._content)
                time.sleep(5)
        except KeyError:
            print(response)
            time.sleep(5)
            continue
    return ''


def swift_call(context: list, system: str, swift_infer_obj):
    query_d: dict = context[-1]
    history_list = context[:-1]

    query: str = query_d['content']
    history_msg = []

    tmp_list = []
    for idx, item in enumerate(history_list):

        if idx % 2 == 0:
            tmp_list.append(item['content'])
        else:
            tmp_list.append(item['content'])
            history_msg.append(tuple(tmp_list))
            tmp_list = []

    try:
        resp_str: str = swift_infer_obj.predict(system=system, query=query, history=history_msg)
    except Exception as e:
        print(e)
        resp_str = ''

    return resp_str


@dataclass
class InferArgs:
    model_name_or_path: str
    model_type: str
    data_path: str
    output_dir: str
    deploy_type: str
    max_new_tokens: int = 2048
    num_infer_samples: int = None


def run_infer(args: InferArgs):

    if args.deploy_type == 'swift':
        from evalscope.third_party.toolbench_static.llm.swift_infer import SwiftInfer, SwiftInferArgs
        swift_infer_args = SwiftInferArgs(
            model_id_or_path=args.model_name_or_path, model_type=args.model_type, max_new_tokens=args.max_new_tokens)
        swift_infer = SwiftInfer(args=swift_infer_args)
    else:
        swift_infer = None

    # load data
    infer_samples = nested_load_test_data(args.data_path)
    if args.num_infer_samples is not None:
        infer_samples = infer_samples[:args.num_infer_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.output_dir, 'predictions.json')):
        with open(os.path.join(args.output_dir, 'predictions.json')) as f:
            processed_samples = json.load(f)
    else:
        processed_samples = []
    preds = []
    refs = []
    for i, o in enumerate(infer_samples):
        if i < len(processed_samples) and 'predictions' in processed_samples[i].keys():
            infer_samples[i]['predictions'] = processed_samples[i]['predictions']
            refs.append(processed_samples[i]['target'])
            preds.append(processed_samples[i]['predictions'])
            continue

        system = o['messages'][0]['content']
        new_msg = o['messages'][1:]

        print('================================')
        print('case', str(i))

        if args.deploy_type == 'minimax':
            response_text = minimax_call(new_msg, system)
        # elif model_args.model_type == 'xingchen':
        #     response_text = spark_call(new_msg, system)
        # elif model_args.model_type == 'xingchen_v2':
        #     response_text = spark_call_v2(new_msg, system, model_args)
        elif args.deploy_type == 'baichuan':
            response_text = baichuan_call(new_msg, system)
        elif args.deploy_type == 'swift':
            assert swift_infer is not None, 'ModelScope Swift infer process is not initialized.'
            response_text = swift_call(new_msg, system, swift_infer)
        else:
            raise NotImplementedError

        candidate = response_text
        print(candidate)
        if candidate.startswith(': '):
            candidate = candidate[2:]
        if candidate.strip() in ['', '.', ',']:
            candidate = 'none'
        reference = infer_samples[i]['target']
        infer_samples[i]['predictions'] = candidate
        if reference.strip() in ['', '.', ',']:
            reference = 'none'
        refs.append(reference)
        preds.append(candidate)

        with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(infer_samples[:i + 1], f, indent=4)

    rouge_l = round(evaluate_rouge_l(preds, refs), 2)
    print('\n*** Overall rouge:', rouge_l)
