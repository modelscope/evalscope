import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

from .spotting_quantization_eval import evaluate_layout_sample as legacy_evaluate_layout_sample
from .spotting_quantization_eval import evaluate_text_sample as legacy_evaluate_text_sample
from .spotting_quantization_eval import parse_all_pairs as legacy_parse_all_pairs

Point = Tuple[float, float]
Quad = List[Point]
SpottingItem = Dict[str, Any]
ExpectedSpottingFormat = Literal['auto', 'legacy', 'v1', 'v2']

CODE_FENCE_RE = re.compile(r'^\s*```[\w+-]*\s*$')
QUOTE_FENCE_RE = re.compile(r'^\s*(?:"""|\'\'\')\s*$')
IMAGE_KEY_RE = re.compile(r'^image_(\d+)$')
FULL_CODE_BLOCK_RE = re.compile(r'^\s*```(?:[\w+-]+)?[ \t]*\n?([\s\S]*?)\n?```\s*$', re.IGNORECASE)


def _ensure_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _strip_outer_wrappers(text: str) -> str:
    lines = text.strip().splitlines()
    while lines and (CODE_FENCE_RE.match(lines[0]) or QUOTE_FENCE_RE.match(lines[0])):
        lines.pop(0)
    while lines and (CODE_FENCE_RE.match(lines[-1]) or QUOTE_FENCE_RE.match(lines[-1])):
        lines.pop()
    return '\n'.join(lines).strip()


def _clean_response_text(text: Any) -> str:
    cleaned = _ensure_text(text).replace('\ufeff', '').strip()
    previous = None
    while cleaned != previous:
        previous = cleaned
        block_match = FULL_CODE_BLOCK_RE.match(cleaned)
        if block_match:
            cleaned = block_match.group(1).strip()
        cleaned = _strip_outer_wrappers(cleaned)
    return cleaned.strip()


def _extract_json_candidate(text: str) -> Optional[str]:
    text = _clean_response_text(text)
    if not text:
        return None

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def _format_coord(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) <= 1e-9:
        return str(int(rounded))
    return (f'{float(value):.6f}').rstrip('0').rstrip('.')


def _bbox_sort_key(item: SpottingItem) -> Tuple[float, float, float, float, str]:
    quad = item['quad']
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return (min(ys), min(xs), max(ys), max(xs), item['text'])


def _sort_items(items: List[SpottingItem]) -> List[SpottingItem]:
    return sorted(items, key=_bbox_sort_key)


def _serialize_items_to_legacy(items: List[SpottingItem]) -> str:
    parts: List[str] = []
    for item in _sort_items(items):
        quad = item['quad']
        coords = ','.join(f'({_format_coord(x)},{_format_coord(y)})' for x, y in quad)
        parts.append(f"<|object_ref_start|>{item['text']}<|object_ref_end|>"
                     f'<|quad_start|>{coords}<|quad_end|>')
    return ''.join(parts)


def _looks_like_legacy(text: str) -> bool:
    return '<|object_ref_start|>' in text or '<|quad_start|>' in text


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError('bool is not a valid coordinate')
    return float(value)


def _normalize_text_field(value: Any) -> str:
    return _ensure_text(value).strip()


def _parse_v1_image_key(key: Any) -> Optional[int]:
    normalized_key = _ensure_text(key).strip().lower()
    if normalized_key == 'image':
        return 0

    if normalized_key.startswith('image'):
        suffix = normalized_key[len('image'):]
        if suffix.isdigit():
            return int(suffix)

    match = IMAGE_KEY_RE.match(normalized_key)
    if match:
        return int(match.group(1))

    return None


def _validate_quad_from_nested(position: Any) -> Quad:
    if not isinstance(position, list) or len(position) != 4:
        raise ValueError('position must be a list of 4 points')

    quad: Quad = []
    for point in position:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('each position point must be [x, y]')
        quad.append((_to_float(point[0]), _to_float(point[1])))
    return quad


def _validate_quad_from_flat(pos: Any) -> Quad:
    if not isinstance(pos, list) or len(pos) != 8:
        raise ValueError('pos must be a flat list of 8 coordinates')

    values = [_to_float(coord) for coord in pos]
    return [
        (values[0], values[1]),
        (values[2], values[3]),
        (values[4], values[5]),
        (values[6], values[7]),
    ]


def _parse_legacy_items(text: str) -> Tuple[List[SpottingItem], bool, Optional[str]]:
    cleaned = _clean_response_text(text)
    if not cleaned:
        return [], True, None

    pairs = legacy_parse_all_pairs(cleaned, normalize=True)
    items = [{'text': pair_text, 'quad': quad} for pair_text, quad in pairs]
    parse_ok = bool(items) or not _looks_like_legacy(cleaned)
    error = None if parse_ok else 'legacy spotting markers found but no valid pairs parsed'
    return items, parse_ok, error


def _parse_v1_items(data: Dict[str, Any]) -> Tuple[List[SpottingItem], bool, Optional[str]]:
    image_entries: List[Tuple[int, Any]] = []
    for key, value in data.items():
        image_index = _parse_v1_image_key(key)
        if image_index is not None:
            image_entries.append((image_index, value))

    if not image_entries:
        return [], False, 'v1 payload missing image/image0/image_* keys'

    items: List[SpottingItem] = []
    errors: List[str] = []
    for _, value in sorted(image_entries, key=lambda item: item[0]):
        if value is None:
            continue
        if not isinstance(value, list):
            errors.append('image_* must be a list')
            continue
        for entry in value:
            if not isinstance(entry, dict):
                errors.append('v1 entry must be an object')
                continue
            try:
                item_text = _normalize_text_field(entry.get('text', ''))
                quad = _validate_quad_from_nested(entry.get('position'))
            except Exception as exc:
                errors.append(str(exc))
                continue
            items.append({'text': item_text, 'quad': quad})

    return items, not errors, '; '.join(errors) if errors else None


def _parse_v2_items(data: Dict[str, Any]) -> Tuple[List[SpottingItem], bool, Optional[str]]:
    images = data.get('images')
    if images is None:
        return [], False, 'v2 payload missing images'
    if not isinstance(images, list):
        return [], False, 'images must be a list'

    items: List[SpottingItem] = []
    errors: List[str] = []
    for image in images:
        if not isinstance(image, dict):
            errors.append('each image must be an object')
            continue
        content = image.get('content', [])
        if content is None:
            continue
        if not isinstance(content, list):
            errors.append('image.content must be a list')
            continue
        for entry in content:
            if not isinstance(entry, dict):
                errors.append('v2 content entry must be an object')
                continue
            entry_type = _ensure_text(entry.get('type', 'text')).strip().lower()
            if entry_type and entry_type != 'text':
                continue
            try:
                item_text = _normalize_text_field(entry.get('text', ''))
                quad = _validate_quad_from_flat(entry.get('pos'))
            except Exception as exc:
                errors.append(str(exc))
                continue
            items.append({'text': item_text, 'quad': quad})

    return items, not errors, '; '.join(errors) if errors else None


def _try_parse_json_payload(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidate = _extract_json_candidate(text)
    if not candidate:
        return None, 'no json object found'

    try:
        parsed = json.loads(candidate)
    except Exception as exc:
        return None, str(exc)

    if not isinstance(parsed, dict):
        return None, 'json payload must be an object'
    return parsed, None


def normalize_spotting_payload(text: Any, expected_format: ExpectedSpottingFormat = 'auto') -> Dict[str, Any]:
    cleaned = _clean_response_text(text)
    if not cleaned:
        return {
            'canonical_text': '',
            'detected_format': expected_format if expected_format != 'auto' else 'empty',
            'parse_ok': True,
            'parse_error': None,
            'items': [],
            'raw_text': cleaned,
        }

    if expected_format == 'legacy':
        items, _, parse_error = _parse_legacy_items(cleaned)
        parse_ok = bool(items)
        if not parse_ok and parse_error is None:
            parse_error = 'expected legacy spotting payload'
        return {
            'canonical_text': _serialize_items_to_legacy(items),
            'detected_format': 'legacy',
            'parse_ok': parse_ok,
            'parse_error': parse_error,
            'items': items,
            'raw_text': cleaned,
        }

    if expected_format in {'v1', 'v2'}:
        payload, json_error = _try_parse_json_payload(cleaned)
        if payload is None:
            return {
                'canonical_text': '',
                'detected_format': expected_format,
                'parse_ok': False,
                'parse_error': json_error or f'expected {expected_format} spotting json payload',
                'items': [],
                'raw_text': cleaned,
            }

        if expected_format == 'v1':
            items, parse_ok, parse_error = _parse_v1_items(payload)
        else:
            items, parse_ok, parse_error = _parse_v2_items(payload)

        # Only fabricate a parse_error message when parsing actually failed.
        # An empty `items` list with `parse_ok=True` is a valid (empty) result.
        if not parse_ok and not parse_error:
            parse_error = f'expected {expected_format} spotting json payload'

        return {
            'canonical_text': _serialize_items_to_legacy(items),
            'detected_format': expected_format,
            'parse_ok': parse_ok and bool(items),
            'parse_error': parse_error,
            'items': items,
            'raw_text': cleaned,
        }

    if _looks_like_legacy(cleaned):
        items, parse_ok, parse_error = _parse_legacy_items(cleaned)
        return {
            'canonical_text': _serialize_items_to_legacy(items),
            'detected_format': 'legacy',
            'parse_ok': parse_ok,
            'parse_error': parse_error,
            'items': items,
            'raw_text': cleaned,
        }

    payload, json_error = _try_parse_json_payload(cleaned)
    if payload is not None:
        if 'images' in payload:
            items, parse_ok, parse_error = _parse_v2_items(payload)
            detected_format = 'v2'
        elif any(_parse_v1_image_key(key) is not None for key in payload.keys()):
            items, parse_ok, parse_error = _parse_v1_items(payload)
            detected_format = 'v1'
        else:
            items = []
            parse_ok = False
            parse_error = 'json payload does not match spotting v1/v2 schema'
            detected_format = 'json_unknown'

        return {
            'canonical_text': _serialize_items_to_legacy(items),
            'detected_format': detected_format,
            'parse_ok': parse_ok,
            'parse_error': parse_error,
            'items': items,
            'raw_text': cleaned,
        }

    items, parse_ok, parse_error = _parse_legacy_items(cleaned)
    return {
        'canonical_text': _serialize_items_to_legacy(items),
        'detected_format': 'unrecognized',
        'parse_ok': parse_ok and bool(items),
        'parse_error': json_error or parse_error or 'unrecognized spotting payload',
        'items': items,
        'raw_text': cleaned,
    }


def evaluate_spotting_sample(
    completion: str,
    solution: str,
    expected_format: ExpectedSpottingFormat = 'auto',
    normalize_text_flag: bool = True,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
    blank_reward: Optional[float] = 0.3,
    pred_to_gt_weight: float = 0.5,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    pred_payload = normalize_spotting_payload(completion, expected_format=expected_format)
    gt_payload = normalize_spotting_payload(solution, expected_format=expected_format)

    layout_metrics = legacy_evaluate_layout_sample(
        completion=pred_payload['canonical_text'],
        solution=gt_payload['canonical_text'],
        normalize_text_flag=normalize_text_flag,
        min_reward=min_reward,
        max_reward=max_reward,
        blank_reward=blank_reward,
        pred_to_gt_weight=pred_to_gt_weight,
        eps=eps,
    )
    text_metrics = legacy_evaluate_text_sample(
        completion=pred_payload['canonical_text'],
        solution=gt_payload['canonical_text'],
    )

    diou_score = float(layout_metrics['diou_score'])
    text_score = float(text_metrics['text_score'])
    overall_score = 0.7 * diou_score + 0.3 * text_score

    return {
        'diou_score': diou_score,
        'text_score': text_score,
        'overall_score': float(overall_score),
        'detected_pred_format': pred_payload['detected_format'],
        'detected_gt_format': gt_payload['detected_format'],
        'pred_parse_ok': bool(pred_payload['parse_ok']),
        'gt_parse_ok': bool(gt_payload['parse_ok']),
        'pred_parse_error': pred_payload['parse_error'],
        'gt_parse_error': gt_payload['parse_error'],
    }


def evaluate_dataset(
    input_file: str,
    normalize_text_flag: bool = True,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
    blank_reward: Optional[float] = 0.3,
    pred_to_gt_weight: float = 0.5,
    eps: float = 1e-9,
    to_percent: bool = True,
) -> 'pd.DataFrame':
    import pandas as pd

    sample_results: List[Dict[str, Any]] = []

    with open(input_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            sample_results.append(
                evaluate_spotting_sample(
                    completion=data.get('completion', ''),
                    solution=data.get('solution', ''),
                    normalize_text_flag=normalize_text_flag,
                    min_reward=min_reward,
                    max_reward=max_reward,
                    blank_reward=blank_reward,
                    pred_to_gt_weight=pred_to_gt_weight,
                    eps=eps,
                )
            )

    if not sample_results:
        result = {'diou_score': 0.0, 'text_score': 0.0, 'overall_score': 0.0}
        return pd.DataFrame([result])

    df_samples = pd.DataFrame(sample_results)
    result = {
        'diou_score': float(df_samples['diou_score'].mean()),
        'text_score': float(df_samples['text_score'].mean()),
        'overall_score': float(df_samples['overall_score'].mean()),
    }

    if to_percent:
        result = {key: value * 100.0 for key, value in result.items()}

    return pd.DataFrame([result])


def main() -> None:
    import argparse
    import os
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True, help='Path to the input JSONL file')
    parser.add_argument('--save_dir', required=True, help='Directory where results will be saved')
    parser.add_argument('--save_name', required=True, help='Output CSV file name')
    parser.add_argument(
        '--pred_to_gt_weight', type=float, default=0.5, help='Weight for the prediction-to-ground-truth direction'
    )
    parser.add_argument('--blank_reward', type=float, default=0.3, help='Reward assigned to blank pages')
    parser.add_argument(
        '--no_percent', action='store_true', help='Keep scores in the 0-1 range instead of converting to percentages'
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)

    df = evaluate_dataset(
        input_file=args.input_file,
        normalize_text_flag=True,
        min_reward=0.0,
        max_reward=1.0,
        blank_reward=args.blank_reward,
        pred_to_gt_weight=args.pred_to_gt_weight,
        eps=1e-9,
        to_percent=not args.no_percent,
    )

    df = df.round(4)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)
    print(df)

    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f'\nResults saved to: {save_path}')


if __name__ == '__main__':
    main()
