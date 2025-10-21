"""
Translate dataset English descriptions to Chinese and persist in description.json.

Behavior:
1. Loads existing description.json (if present).
2. Iterates all registered DataAdapters.
3. For each dataset:
   - If not in JSON OR stored English != current English, re-translate.
   - Otherwise reuse stored zh.
4. Writes back description.json with structure:
   {
     "<dataset_name>": {
        "en": "...",
        "zh": "...",
        "updated_at": "2024-01-01T12:00:00Z"
     },
     ...
   }

Single function translate_description(description) kept for unit use.
"""
import os

os.environ['BUILD_DOC'] = '1'  # To avoid some heavy dependencies
import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import dotenv_values
from openai import OpenAI
from tqdm import tqdm
from typing import Any, Dict

from . import DESCRIPTION_JSON_PATH, load_description_json, save_description_json

env = dotenv_values('.env')


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=env.get('DASHSCOPE_API_KEY'),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )


def translate_description(description: str, client: OpenAI | None = None) -> str:
    # Reuse provided client to avoid re-instantiation in threads
    if client is None:
        client = _get_client()
    completion = client.chat.completions.create(
        model='qwen-plus',
        messages=[
            {'role': 'system', 'content': ('You are a helpful assistant. '
                                           'Translate the following English description to Chinese. '
                                           'Without changing the meaning, make it concise and fluent. '
                                           'Return ONLY the translated text. Keep markdown formatting unchanged.')},
            {'role': 'user', 'content': description}],
    )
    return completion.choices[0].message.content.strip()

def convert_link(text: str) -> str:
    """Convert markdown links from English to Chinese."""
    return text.replace(
        'https://evalscope.readthedocs.io/en/',
        'https://evalscope.readthedocs.io/zh-cn/'
    )

def update_all_descriptions(force: bool = False,
                            dry_run: bool = False,
                            workers: int = 4) -> Dict[str, Any]:
    """
    Update / create description.json by traversing all benchmarks.
    force=True will retranslate all.
    dry_run=True will not write file.
    limit: limit number of (re)translations this run.
    """
    from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
    data = load_description_json(DESCRIPTION_JSON_PATH)
    now_iso = datetime.datetime.now().replace(microsecond=0).isoformat() + 'Z'

    to_translate: list[tuple[str, str]] = []
    skipped = 0

    # Collect candidates
    for benchmark in BENCHMARK_REGISTRY.values():
        adapter = get_benchmark(benchmark.name)
        name = adapter.name
        en_text = (adapter.description or '').strip()
        if not en_text:
            skipped += 1
            continue
        record = data.get(name)
        needs = force or record is None or record.get('en', '').strip() != en_text or not record.get('zh', '').strip()
        if not needs:
            skipped += 1
            continue
        to_translate.append((name, en_text))

    translated = 0
    client = _get_client()

    if to_translate:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {
                executor.submit(translate_description, en_txt, client): (name, en_txt)
                for name, en_txt in to_translate
            }
            for future in tqdm(as_completed(future_map),
                               total=len(future_map),
                               desc='Translating datasets',
                               unit='ds'):
                name, en_txt = future_map[future]
                try:
                    zh_txt = future.result()
                except Exception as e:
                    zh_txt = ''  # On failure leave empty so can retry later
                    print(f'Error translating {name}: {e}')
                
                zh_txt = convert_link(zh_txt)
                data[name] = {
                    'en': en_txt,
                    'zh': zh_txt,
                    'updated_at': now_iso
                }
                translated += 1

    if translated and not dry_run:
        save_description_json(data, DESCRIPTION_JSON_PATH)

    return {
        'total': translated + skipped,
        'translated': translated,
        'skipped': skipped,
        'workers': workers,
        'written': bool(translated and not dry_run),
        'path': DESCRIPTION_JSON_PATH
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Translate / update dataset descriptions.')
    parser.add_argument('--force', action='store_true', help='Retranslate all even if unchanged.')
    parser.add_argument('--dry-run', action='store_true', help='Do not write file.')
    parser.add_argument('--workers', type=int, default=4, help='Thread pool size.')
    args = parser.parse_args()
    result = update_all_descriptions(force=args.force,
                                     dry_run=args.dry_run,
                                     workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

