"""Expose AutoTrust's calibrated decision readout over a standard vLLM completions server.

The bare prompt, verbalizer bias and per-kind temperatures follow the released
autotrust/JEV-27B implementation (Apache-2.0). No vLLM internal imports are used.
"""

import argparse
import json
import math
import string
from pathlib import Path
from typing import Any


def main() -> None:
    """Read every candidate token and apply the released head's calibration."""
    import httpx
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--upstream', default='http://127.0.0.1:18132/v1')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18133)
    args = parser.parse_args()
    head = json.loads((args.model_path / 'adapter_vllm/decision_head.json').read_text())
    temperatures = json.loads((args.model_path / 'calibration.json').read_text())['per_kind']
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    labels = []
    for label in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) == 1 and ids[0] in tokenizer.encode(f'x\n{label}) y', add_special_tokens=False):
            labels.append((label, ids[0]))
        if len(labels) == 256:
            break
    lo, hi = head['slots']['ranges']['choice']
    if [token for _, token in labels[:hi - lo]] != head['verbalizer_ids'][lo:hi]:
        raise ValueError('Tokenizer does not match the released decision head.')
    client = httpx.AsyncClient(timeout=180)
    app = FastAPI()

    @app.get('/health')
    async def health() -> dict:
        response = await client.get(args.upstream.removesuffix('/v1') + '/health')
        response.raise_for_status()
        return {'status': 'ready', 'model': 'autotrust/JEV-27B', 'backend': 'vllm_completions'}

    @app.post('/v1/decide')
    async def decide(body: dict[str, Any]) -> dict:
        kind = body.get('kind')
        if kind not in ('choice', 'noul', 'score') or body.get('thinking', 'off') != 'off':
            raise HTTPException(422, 'Only choice/noul/score System One decisions with thinking=off are supported.')
        options = body.get('options', []) if kind == 'choice' else ['false', 'true'] if kind == 'noul' else [str(i) for i in range(6)]
        if not 2 <= len(options) <= len(labels):
            raise HTTPException(422, 'Unsupported option count.')
        lo, hi = head['slots']['ranges'][kind]
        ids = [token for _, token in labels[:len(options)]] if kind == 'choice' else head['verbalizer_ids'][lo:hi]
        biases = [head['bias'][lo + i] if lo + i < hi else 0.0 for i in range(len(options))]
        lines = [f'{labels[i][0]}) {option}' for i, option in enumerate(options)] if kind == 'choice' else options
        state = body['state'] if isinstance(body['state'], str) else json.dumps(body['state'], ensure_ascii=False)
        prompt = f'[kind] {kind}\n[state] {state}\n[question] {body["question"]}\n[options]\n' + '\n'.join(lines) + '\n[decision]:'
        response = await client.post(args.upstream + '/completions', json={
            'model': 'jev-decision', 'prompt': prompt, 'max_tokens': 1, 'temperature': 1.0,
            'top_k': 0, 'top_p': 1.0, 'min_p': 0.0, 'repetition_penalty': 1.0,
            'logprobs': len(ids), 'allowed_token_ids': ids, 'add_special_tokens': False,
            'return_tokens_as_token_ids': True,
        })
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.text)
        raw = response.json()
        logprobs = {int(key.split(':')[1]): value for key, value in
                    raw['choices'][0]['logprobs']['top_logprobs'][0].items()}
        if any(token not in logprobs or not math.isfinite(logprobs[token]) for token in ids):
            raise HTTPException(502, 'vLLM did not return finite logprobs for every candidate token.')
        z = [(logprobs[token] + bias) / temperatures[kind] for token, bias in zip(ids, biases)]
        exp = [math.exp(value - max(z)) for value in z]
        probabilities = [value / sum(exp) for value in exp]
        index = max(range(len(options)), key=probabilities.__getitem__)
        return {'model': 'autotrust/JEV-27B', 'kind': kind, 'effective_kind': kind,
                'options': options, 'probabilities': probabilities, 'choice_index': index, 'choice': options[index],
                'usage': raw.get('usage', {}), 'protocol': 'jev27-bare-v1', 'num_model_requests': 1}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
