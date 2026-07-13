import zipfile
from typing import Any, Dict, Optional


def optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid MVBench {field_name}: {value!r}') from exc


def format_seconds(value: float) -> str:
    return f'{value:g}'


def format_time_range(start: Optional[float], end: Optional[float]) -> str:
    if start is None and end is None:
        return ''
    if start is not None and end is not None:
        return f'from {format_seconds(start)}s to {format_seconds(end)}s'
    if start is not None:
        return f'after {format_seconds(start)}s'
    return f'before {format_seconds(end)}s'


def format_video_context(record: Dict[str, Any], start: Optional[float], end: Optional[float]) -> str:
    context_parts = []
    time_range = format_time_range(start, end)
    if time_range:
        context_parts.append(f'Answer based on the video segment {time_range}.')

    subtitle = record.get('subtitle') or record.get('subtitles')
    if subtitle:
        context_parts.append(f'Subtitles:\n{subtitle}')

    return '\n'.join(context_parts)


def build_question(record: Dict[str, Any], start: Optional[float], end: Optional[float]) -> str:
    context = format_video_context(record, start=start, end=end)
    question = str(record['question'])
    if not context:
        return question
    return f'{context}\n\n{question}'


def find_archive_member(archive_path: str, subset: str, video_name: str) -> str:
    normalized_subset = subset.replace('_', '').lower()
    with zipfile.ZipFile(archive_path) as zip_file:
        matches = [
            name for name in zip_file.namelist()
            if not name.endswith('/') and (name.endswith(f'/{video_name}') or name.endswith(video_name))
        ]
    if not matches:
        raise FileNotFoundError(f'Video {video_name} was not found in archive {archive_path}.')

    preferred = [name for name in matches if normalized_subset in name.replace('_', '').lower()]
    return sorted(preferred or matches)[0]
