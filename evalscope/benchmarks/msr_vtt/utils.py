import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional

DEFAULT_PROMPT = 'Describe the video in one concise sentence.'


def optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {field_name} value: {value!r}') from exc


def extract_captions(record: Dict[str, Any]) -> List[str]:
    for field in ('references', 'caption', 'captions'):
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
    return []


def unique_texts(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def group_caption_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge annotation rows sharing a ``video_id`` into one record with unique references."""
    grouped: 'OrderedDict[str, Dict[str, Any]]' = OrderedDict()
    for record in records:
        video_id = str(record.get('video_id') or record.get('id') or len(grouped))
        captions = extract_captions(record)
        if video_id not in grouped:
            grouped[video_id] = copy.deepcopy(record)
            grouped[video_id]['references'] = []
        grouped[video_id]['references'].extend(captions)

    for record in grouped.values():
        record['references'] = unique_texts(record['references'])
    return list(grouped.values())
