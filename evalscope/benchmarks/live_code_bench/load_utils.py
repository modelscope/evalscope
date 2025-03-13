import base64
import json
import pickle
import zlib
from datetime import datetime

from evalscope.benchmarks.live_code_bench.prompts import CodeGenerationPromptConstants
from evalscope.utils.logger import get_logger

logger = get_logger()


def transform(item):
    # Define the dataitem mapping logic

    # starter_code
    if item['starter_code']:
        format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'  # noqa: E501
        format_prompt += f"```python\n{item['starter_code']}\n```\n\n"
    else:
        format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n'  # noqa: E501
        format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'

    item['format_prompt'] = format_prompt

    # load test cases
    public_test_cases = item['public_test_cases']
    public_test_cases = json.loads(item['public_test_cases'])

    private_test_cases = item['private_test_cases']
    try:
        private_test_cases = json.loads(item['private_test_cases'])
    except Exception as e:  # noqa: F841
        private_test_cases = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(private_test_cases.encode('utf-8'))  # type: ignore
                                         )))  # type: ignore

    # load metadata
    metadata = json.loads(item['metadata'])
    evaluation_sample = json.dumps({
        'inputs': [t['input'] for t in public_test_cases + private_test_cases],
        'outputs': [t['output'] for t in public_test_cases + private_test_cases],
        'fn_name': metadata.get('func_name', None),
    })
    item['evaluation_sample'] = evaluation_sample

    return item


def filter_date(dataset, start_date=None, end_date=None):
    new_dataset = []

    for item in dataset:
        contest_date = datetime.fromisoformat(item['contest_date'])
        if start_date is not None:
            p_start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if p_start_date > contest_date:
                continue

        if end_date is not None:
            p_end_date = datetime.strptime(end_date, '%Y-%m-%d')
            if p_end_date < contest_date:
                continue

        new_dataset.append(item)

    if start_date or end_date:
        logger.info(
            f'Filtered dataset with start_date: {start_date}, end_date: {end_date}, remaining items: {len(new_dataset)}'
        )
    return new_dataset
