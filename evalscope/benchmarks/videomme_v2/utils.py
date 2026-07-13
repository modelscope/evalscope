import json
import re
import zipfile
from typing import Any, Dict, List, Optional

OPTION_PATTERN = re.compile(r'^\s*([A-Z])[\.\)]\s*(.*)$')


def parse_options(options: Any) -> List[str]:
    if isinstance(options, list):
        choices = [str(option).strip() for option in options]
    else:
        choices = []
        for line in str(options).splitlines():
            line = line.strip()
            if not line:
                continue
            match = OPTION_PATTERN.match(line)
            if match:
                choices.append(match.group(2).strip())
            elif choices:
                choices[-1] = f'{choices[-1]} {line}'
            else:
                choices.append(line)
    if not choices:
        raise ValueError(f'Invalid Video-MME-v2 options: {options!r}')
    return choices


def normalize_answer(answer: Any, choices: List[str]) -> str:
    target = str(answer).strip().upper()
    valid_answers = {chr(ord('A') + idx) for idx in range(len(choices))}
    if target not in valid_answers:
        raise ValueError(f'Invalid Video-MME-v2 answer {answer!r}; expected one of {sorted(valid_answers)}.')
    return target


def normalize_video_id(video_id: str) -> str:
    if not re.fullmatch(r'\d{1,3}', str(video_id)):
        raise ValueError(f'Invalid Video-MME-v2 video id: {video_id}')
    video_num = int(video_id)
    if not 1 <= video_num <= 800:
        raise ValueError(f'Invalid Video-MME-v2 video id: {video_id}')
    return f'{video_num:03d}'


def archive_name(video_id: str) -> str:
    video_num = int(normalize_video_id(video_id))
    archive_num = ((video_num - 1) // 20) + 1
    return f'{archive_num:03d}.zip'


def find_archive_member(archive_path: str, video_id: str) -> str:
    expected = f'{normalize_video_id(video_id)}.mp4'
    with zipfile.ZipFile(archive_path) as zip_file:
        matches = [
            name for name in zip_file.namelist()
            if not name.endswith('/') and (name.endswith(f'/{expected}') or name.endswith(expected))
        ]
    if not matches:
        raise FileNotFoundError(f'Video {expected} was not found in archive {archive_path}.')
    return sorted(matches)[0]


def subtitle_text_from_jsonl(raw_text: str, word_limit: int) -> str:
    words = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        text = str(item.get('text') or '').strip()
        if text:
            words.append(text)
            if word_limit and len(words) >= word_limit:
                break
    return ' '.join(words)


def build_question(record: Dict[str, Any], subtitle: Optional[str]) -> str:
    question = str(record['question'])
    if not subtitle:
        return question
    return f'Subtitles:\n{subtitle}\n\n{question}'
