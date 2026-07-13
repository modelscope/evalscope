import os
import zipfile
from typing import Any, Dict, Optional


def optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid TVBench {field_name}: {value!r}') from exc


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


def format_video_context(start: Optional[float], end: Optional[float]) -> str:
    time_range = format_time_range(start, end)
    if not time_range:
        return ''
    return f'Answer based on the video segment {time_range}.'


def build_question(record: Dict[str, Any], start: Optional[float], end: Optional[float]) -> str:
    context = format_video_context(start=start, end=end)
    question = str(record['question'])
    if not context:
        return question
    return f'{context}\n\n{question}'


def safe_join(root_dir: str, *path_parts: str) -> str:
    output_path = os.path.abspath(os.path.join(root_dir, *path_parts))
    if os.path.commonpath([root_dir, output_path]) != root_dir:
        raise ValueError(f'Invalid TVBench video path: {os.path.join(*path_parts)}')
    return output_path


def archive_path(subset: str, video_name: str) -> str:
    if subset == 'action_antonym':
        raise FileNotFoundError(
            'The TVBench repository does not expose video/action_antonym.zip. '
            'Please configure extra_params.video_dir with local action_antonym AVI files.'
        )
    if subset == 'egocentric_sequence':
        archive_name = video_name.split('/', 1)[0]
        if not archive_name or archive_name == video_name:
            raise ValueError(f'Invalid TVBench egocentric video path: {video_name!r}')
        return f'video/egocentric_sequence/{archive_name}.zip'
    return f'video/{subset}.zip'


def find_archive_member(zip_file: zipfile.ZipFile, video_name: str) -> str:
    normalized_video_name = video_name.replace('\\', '/').lstrip('/')
    video_basename = os.path.basename(normalized_video_name)
    member_names = [name for name in zip_file.namelist() if not name.endswith('/')]
    exact_matches = [
        name for name in member_names if name.replace('\\', '/').endswith(f'/{normalized_video_name}')
        or name.replace('\\', '/') == normalized_video_name
    ]
    if exact_matches:
        return sorted(exact_matches)[0]

    basename_matches = [name for name in member_names if os.path.basename(name.replace('\\', '/')) == video_basename]
    if basename_matches:
        return sorted(basename_matches)[0]
    raise FileNotFoundError(f'Video {video_name} was not found in archive.')
