# Copyright (c) Alibaba, Inc. and its affiliates.
import re
import string
import unicodedata
from collections import Counter
from nltk.stem import PorterStemmer
from typing import Any, Dict, List, Tuple

DATA_FILE = 'locomo10.json'
CATEGORY_IDS = [4, 1, 2, 3, 5]
CATEGORY_NAMES = {
    1: 'multi_hop',
    2: 'temporal',
    3: 'commonsense',
    4: 'single_hop',
    5: 'adversarial',
}
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)
STEMMER = PorterStemmer()

CONV_START_PROMPT = (
    'Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and '
    'the date of each conversation is wriiten at the beginning of the conversation.\n\n'
)

QA_PROMPT = (
    '\nBased on the above context, write an answer in the form of a short phrase for the following question. '
    'Answer with exact words from the context whenever possible.\n\nQuestion: {} Short answer:'
)

QA_PROMPT_CAT_5 = '\nBased on the above context, answer the following question.\n\nQuestion: {} Short answer:'


def build_qa_prompt(conversation: Dict[str, Any], qa: Dict[str, Any], eval_mode: str) -> Tuple[str, Dict[str, Any]]:
    """Build the LoCoMo QA prompt and return prompt metadata."""
    question = _build_question(qa)
    history = _build_history(conversation, evidence=qa.get('evidence', []), eval_mode=eval_mode)
    speaker_a = conversation.get('speaker_a') or _infer_speakers(conversation)[0]
    speaker_b = conversation.get('speaker_b') or _infer_speakers(conversation)[1]
    prompt = CONV_START_PROMPT.format(speaker_a, speaker_b) + history
    if qa.get('category') == 5:
        prompt += QA_PROMPT_CAT_5.format(question)
    else:
        prompt += QA_PROMPT.format(question)
    return prompt, {'question': question, 'evidence': qa.get('evidence', [])}


def get_target_answer(qa: Dict[str, Any]) -> str:
    """Return the scoring target for a LoCoMo QA item."""
    if qa.get('category') == 5 and 'answer' not in qa:
        return 'No information available'
    return str(qa.get('answer', ''))


def locomo_f1_score(prediction: str, reference: str, category: int) -> float:
    """Compute LoCoMo's QA score for one prediction."""
    if category == 5:
        lowered = prediction.lower()
        if 'no information available' in lowered or 'not mentioned' in lowered:
            return 1.0
        return 0.0
    if category == 1:
        return _multi_answer_f1(prediction, reference)
    if category == 3:
        reference = reference.split(';')[0].strip()
    return _token_f1(prediction, reference)


def _build_question(qa: Dict[str, Any]) -> str:
    question = str(qa['question'])
    if qa.get('category') == 2:
        return question + ' Use DATE of CONVERSATION to answer with an approximate date.'
    if qa.get('category') == 5:
        return question + " If the answer is not mentioned in the conversation, answer 'No information available'."
    return question


def _build_history(conversation: Dict[str, Any], evidence: List[str], eval_mode: str) -> str:
    session_nums = _session_numbers(conversation)
    evidence_set = set(evidence or [])
    history = ''
    for session_num in session_nums:
        session_key = f'session_{session_num}'
        turns = conversation.get(session_key) or []
        if eval_mode == 'oracle_context':
            turns = [turn for turn in turns if turn and turn.get('dia_id') in evidence_set]
            if not turns:
                continue
        date_time = conversation.get(f'{session_key}_date_time') or ''
        history += f'DATE: {date_time}\nCONVERSATION:\n'
        for turn in turns:
            history += _format_turn(turn)
        history += '\n'
    return history


def _format_turn(turn: Dict[str, Any]) -> str:
    text = f'{turn.get("speaker", "")} said, "{turn.get("text", "")}"'
    if turn.get('blip_caption'):
        text += f' and shared {turn["blip_caption"]}.'
    return text + '\n\n'


def _session_numbers(conversation: Dict[str, Any]) -> List[int]:
    nums = []
    for key in conversation:
        if key.startswith('session_') and not key.endswith('_date_time'):
            suffix = key.split('_')[-1]
            if suffix.isdigit():
                nums.append(int(suffix))
    return sorted(nums)


def _infer_speakers(conversation: Dict[str, Any]) -> Tuple[str, str]:
    speakers = []
    for session_num in _session_numbers(conversation):
        for turn in conversation.get(f'session_{session_num}') or []:
            speaker = turn.get('speaker')
            if speaker and speaker not in speakers:
                speakers.append(speaker)
            if len(speakers) == 2:
                return speakers[0], speakers[1]
    return '', ''


def _normalize_answer(text: str) -> str:
    text = unicodedata.normalize('NFD', str(text)).lower().replace(',', '')
    text = text.translate(PUNCTUATION_TABLE)
    text = re.sub(r'\b(a|an|the|and)\b', ' ', text)
    return ' '.join(text.split())


def _stem(word: str) -> str:
    return STEMMER.stem(word)


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = [_stem(token) for token in _normalize_answer(prediction).split()]
    reference_tokens = [_stem(token) for token in _normalize_answer(reference).split()]
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(reference_tokens) if reference_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _multi_answer_f1(prediction: str, reference: str) -> float:
    predictions = [p.strip() for p in prediction.split(',')]
    references = [r.strip() for r in reference.split(',')]
    if not references:
        return 0.0
    return sum(max(_token_f1(pred, ref) for pred in predictions) for ref in references) / len(references)
