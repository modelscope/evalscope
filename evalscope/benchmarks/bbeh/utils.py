# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapted from https://github.com/google-deepmind/bbeh/blob/main/bbeh/evaluate.py."""


def extract_answer(sample: str) -> str:
    """Extract and normalize the answer in the official order."""
    answer = sample.strip()
    for prefix in ('The answer is:', 'The final answer is ', 'The final answer is: ', 'The answer is '):
        if prefix in answer:
            answer = answer.rsplit(prefix, 1)[-1].strip()
    answer = answer.removesuffix('.')
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer[1:-1]
    for marker in ('boxed{', 'text{', 'texttt{'):
        if marker in answer and answer.endswith('}'):
            answer = answer[:-1].split(marker)[1]
    return answer.lower().replace(', ', ',').replace('**', '').split('\n')[0].removesuffix('.')


def match_answer(prediction: str, reference: str) -> bool:
    """Match an extracted answer against a reference using the official rules."""
    reference = reference.strip().lower().replace(', ', ',')
    if prediction == reference:
        return True

    # Parenthesized letters return immediately, even when another rule could match.
    if len(prediction) == 3 and prediction.startswith('(') and prediction.endswith(')'):
        return prediction[1] == reference
    if len(reference) == 3 and reference.startswith('(') and reference.endswith(')'):
        return reference[1] == prediction

    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    return (
        prediction.replace("'", '') == reference.replace("'", '')
        or prediction == f'[{reference}]'
        or reference == f'[{prediction}]'
        or prediction.removesuffix('?') == reference
    )
