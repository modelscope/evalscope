# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import json
from typing import Any, Dict, List, Tuple

QUESTION_TYPES = [
    'single-session-user',
    'single-session-preference',
    'single-session-assistant',
    'multi-session',
    'temporal-reasoning',
    'knowledge-update',
]

SUBSET_TO_FILE = {
    'oracle': 'longmemeval_oracle.json',
    's': 'longmemeval_s_cleaned.json',
    'm': 'longmemeval_m_cleaned.json',
}

LONGMEMEVAL_PROMPT_DIRECT = (
    'I will give you several history chats between you and a user. Please answer the question based on the '
    'relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'
)

LONGMEMEVAL_PROMPT_CON = (
    'I will give you several history chats between you and a user. Please answer the question based on the '
    'relevant chat history. Answer the question step by step: first extract all the relevant information, and '
    'then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n'
    'Question: {}\nAnswer (step by step):'
)


def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """Build the official LongMemEval QA judge prompt."""
    if abstention:
        template = (
            'I will give you an unanswerable question, an explanation, and a response from a model. Please answer '
            'yes if the model correctly identifies the question as unanswerable. The model could say that the '
            'information is incomplete, or some other information is given but the asked information is not.\n\n'
            'Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the '
            'question as unanswerable? Answer yes or no only.'
        )
        return template.format(question, answer, response)

    if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
        template = (
            'I will give you a question, a correct answer, and a response from a model. Please answer yes if the '
            'response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct '
            'answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If '
            'the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: '
            '{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.'
        )
        return template.format(question, answer, response)

    if task == 'temporal-reasoning':
        template = (
            'I will give you a question, a correct answer, and a response from a model. Please answer yes if the '
            'response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct '
            'answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If '
            'the response only contains a subset of the information required by the answer, answer no. In addition, '
            'do not penalize off-by-one errors for the number of days. If the question asks for the number of '
            'days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer '
            'is 18), the model\'s response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel '
            'Response: {}\n\nIs the model response correct? Answer yes or no only.'
        )
        return template.format(question, answer, response)

    if task == 'knowledge-update':
        template = (
            'I will give you a question, a correct answer, and a response from a model. Please answer yes if the '
            'response contains the correct answer. Otherwise, answer no. If the response contains some previous '
            'information along with an updated answer, the response should be considered as correct as long as the '
            'updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n'
            'Is the model response correct? Answer yes or no only.'
        )
        return template.format(question, answer, response)

    if task == 'single-session-preference':
        template = (
            'I will give you a question, a rubric for desired personalized response, and a response from a model. '
            'Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does '
            'not need to reflect all the points in the rubric. The response is correct as long as it recalls and '
            'utilizes the user\'s personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: '
            '{}\n\nIs the model response correct? Answer yes or no only.'
        )
        return template.format(question, answer, response)

    raise NotImplementedError(f'Unsupported LongMemEval question type: {task}')


def build_generation_prompt(
    entry: Dict[str, Any],
    eval_mode: str,
    topk_context: int,
    history_format: str,
    user_only: bool,
    reading_method: str,
    retriever_type: str,
) -> Tuple[str, Dict[str, Any]]:
    """Build a LongMemEval generation prompt and return retrieval metadata."""
    if reading_method == 'direct':
        answer_prompt_template = LONGMEMEVAL_PROMPT_DIRECT
    elif reading_method == 'con':
        answer_prompt_template = LONGMEMEVAL_PROMPT_CON
    else:
        raise ValueError(f'Unsupported reading_method: {reading_method}')

    chunks = _select_chunks(
        entry=entry,
        eval_mode=eval_mode,
        topk_context=topk_context,
        user_only=user_only,
        retriever_type=retriever_type,
    )
    chunks.sort(key=lambda x: x[0])
    history_string = _format_history(chunks, history_format=history_format)
    prompt = answer_prompt_template.format(history_string, entry['question_date'], entry['question'])
    metadata = {'retrieved_ids': [item[1] for item in chunks]}
    return prompt, metadata


def _select_chunks(
    entry: Dict[str, Any],
    eval_mode: str,
    topk_context: int,
    user_only: bool,
    retriever_type: str,
) -> List[Tuple[str, str, Any]]:
    if eval_mode in ['long_context', 'oracle_context']:
        chunks = [(date, session_id, _filter_user_only(session, user_only)) for date, session_id, session in
                  zip(entry['haystack_dates'], entry['haystack_session_ids'], entry['haystack_sessions'])]
        return chunks[-topk_context:]

    if eval_mode != 'retrieval_log':
        raise ValueError(f'Unsupported eval_mode: {eval_mode}')

    if retriever_type not in ['flat-session', 'flat-turn']:
        raise ValueError(
            f'retriever_type must be flat-session or flat-turn in retrieval_log mode, got: {retriever_type}'
        )

    corpusid2date, corpusid2entry = _build_corpus_maps(entry)
    chunks = []
    ranked_items = entry.get('retrieval_results', {}).get('ranked_items', [])
    for ret_result_entry in ranked_items[:topk_context]:
        corpus_id = ret_result_entry['corpus_id'].replace('noans_', 'answer_')
        if retriever_type == 'flat-session':
            if corpus_id not in corpusid2entry:
                continue
            chunks.append(
                (corpusid2date[corpus_id], corpus_id, _filter_user_only(corpusid2entry[corpus_id], user_only))
            )
        else:
            session_id = '_'.join(corpus_id.split('_')[:-1])
            try:
                turn_id = int(corpus_id.split('_')[-1]) - 1
            except ValueError:
                continue
            if session_id not in corpusid2entry:
                continue
            session = corpusid2entry[session_id]
            if turn_id < 0 or turn_id >= len(session):
                continue
            round_data = [session[turn_id]]
            next_turn_id = turn_id + 1
            if next_turn_id < len(session):
                round_data.append(session[next_turn_id])
            round_data = _filter_user_only(round_data, user_only)
            if round_data:
                chunks.append((corpusid2date[session_id], corpus_id, round_data))
    return chunks


def _build_corpus_maps(entry: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    corpusid2date, corpusid2entry = {}, {}
    for session_date, session_id, session_entry in zip(
        entry['haystack_dates'], entry['haystack_session_ids'], entry['haystack_sessions']
    ):
        corpusid2date[session_id] = session_date
        corpusid2entry[session_id] = session_entry
        for i_turn, turn_entry in enumerate(session_entry):
            turn_id = f'{session_id}_{i_turn + 1}'
            corpusid2date[turn_id] = session_date
            corpusid2entry[turn_id] = turn_entry
    return corpusid2date, corpusid2entry


def _filter_user_only(session: Any, user_only: bool) -> Any:
    cleaned = _remove_has_answer(session)
    if not user_only or not isinstance(cleaned, list):
        return cleaned
    return [turn for turn in cleaned if isinstance(turn, dict) and turn.get('role') == 'user']


def _remove_has_answer(value: Any) -> Any:
    if isinstance(value, list):
        return [_remove_has_answer(item) for item in value]
    if isinstance(value, dict):
        return {key: _remove_has_answer(item) for key, item in value.items() if key != 'has_answer'}
    return value


def _format_history(chunks: List[Tuple[str, str, Any]], history_format: str) -> str:
    history_string = ''
    for idx, (date, _corpus_id, chunk) in enumerate(chunks, start=1):
        if history_format == 'json':
            session_string = '\n' + json.dumps(chunk, ensure_ascii=False)
        elif history_format == 'nl':
            session_string = _format_nl(chunk)
        else:
            raise ValueError(f'Unsupported history_format: {history_format}')
        history_string += f'\n### Session {idx}:\nSession Date: {date}\nSession Content:\n{session_string}\n'
    return history_string


def _format_nl(chunk: Any) -> str:
    if isinstance(chunk, list):
        return ''.join([
            f"\n\n{turn.get('role', '')}: {str(turn.get('content', '')).strip()}" for turn in chunk
            if isinstance(turn, dict)
        ])
    if isinstance(chunk, dict):
        return f"{chunk.get('role', '')}: {str(chunk.get('content', '')).strip()}"
    return str(chunk)
