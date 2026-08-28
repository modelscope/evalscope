import base64
import io
import json
import re
from typing import Any, List, Optional, Sequence, Tuple

_FENCE_RE = re.compile(r'```(?:json)?\s*(.*?)```', flags=re.DOTALL | re.IGNORECASE)
_PRIORITY_KEYS = ('bboxes', 'boxes', 'bbox', 'bbox_2d', 'predictions', 'results', 'objects', 'coordinates')


def _load_last_json_candidate(text: str) -> Optional[Any]:
    """Load the last fenced JSON value, or an unfenced JSON value ending the reply."""
    fenced_chunks = [match.group(1).strip() for match in _FENCE_RE.finditer(text) if match.group(1).strip()]
    for chunk in reversed(fenced_chunks):
        try:
            return json.loads(chunk)
        except (json.JSONDecodeError, TypeError):
            continue

    stripped = text.rstrip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in '[{':
            continue
        try:
            value, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if index + end == len(stripped):
            return value
    return None


def _collect_boxes(value: Any, boxes: List[List[float]], max_boxes: int = 100) -> None:
    if len(boxes) >= max_boxes:
        return
    if isinstance(value, list):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            boxes.append([float(item) for item in value])
            return
        for item in value:
            _collect_boxes(item, boxes, max_boxes)
        return
    if not isinstance(value, dict):
        return

    for key in _PRIORITY_KEYS:
        if key in value:
            _collect_boxes(value[key], boxes, max_boxes)
    for key, item in value.items():
        if key not in _PRIORITY_KEYS:
            _collect_boxes(item, boxes, max_boxes)


def to_normalized_xyxy(box: Sequence[float], image_size: Sequence[int], box_format: str) -> List[float]:
    """Convert an official Ref-Adv bbox format to sorted, clipped unit coordinates."""
    x1, y1, x2, y2 = (float(value) for value in box)
    width, height = image_size
    if box_format == 'abs_xyxy':
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    elif box_format == 'norm_1000_xyxy':
        x1, y1, x2, y2 = (value / 1000.0 for value in (x1, y1, x2, y2))
    elif box_format != 'norm_1_xyxy':
        raise ValueError(f'Unsupported pred_box_format: {box_format}')

    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return [
        min(max(left, 0.0), 1.0),
        min(max(top, 0.0), 1.0),
        min(max(right, 0.0), 1.0),
        min(max(bottom, 0.0), 1.0),
    ]


def parse_bboxes(
    text: str,
    image_size: Sequence[int],
    box_format: str,
) -> Tuple[List[List[float]], str]:
    """Parse and normalize bounding boxes using the official Ref-Adv JSON rules."""
    if not text or not text.strip():
        return [], 'empty_response'

    parsed = _load_last_json_candidate(text)
    if parsed is None:
        return [], 'no_bbox_found'

    raw_boxes: List[List[float]] = []
    _collect_boxes(parsed, raw_boxes)
    if not raw_boxes:
        return [], 'no_bbox_found'

    boxes: List[List[float]] = []
    seen = set()
    for box in raw_boxes:
        normalized = to_normalized_xyxy(box, image_size, box_format)
        key = tuple(round(value, 6) for value in normalized)
        if key in seen:
            continue
        seen.add(key)
        boxes.append(normalized)
    return boxes, ''


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate intersection over union for two xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def distractor_bin(count: int) -> Optional[str]:
    """Return the official distractor-count reporting bin."""
    if 2 <= count <= 3:
        return '2-3'
    if 4 <= count <= 6:
        return '4-6'
    if count >= 7:
        return '>=7'
    return None


def base64_image_size(data_uri: str) -> Tuple[int, int]:
    """Return the width and height of a base64 data-URI image."""
    from PIL import Image

    payload = data_uri.split(',', 1)[-1]
    with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
        return image.size
