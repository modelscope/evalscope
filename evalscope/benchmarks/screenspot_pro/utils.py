"""Prediction parsing and scoring helpers for the ScreenSpot-Pro benchmark."""
import base64
import io
import re
from typing import List, Optional, Sequence, Tuple

# A number token, optionally negative, fractional or percentage-suffixed.
_NUM_TOKEN = r'-?\d+(?:\.\d+)?%?'
_BBOX_KEYWORDS = ('bbox', 'box', 'rect', 'rectangle')

# Many VLMs answer "normalized" coordinates scaled to a 0-1000 grid instead of [0, 1].
_THOUSANDTHS_SCALE = 1000.0

# The prompt asks for the final answer after an ``Answer:`` marker, so the point can be
# located exactly instead of being guessed from the surrounding reasoning.
_ANSWER_MARKER_RE = re.compile(r'(?:final\s+)?answer\s*[:：]', re.IGNORECASE)

AUTO = 'auto'
NORMALIZED = 'normalized'
THOUSANDTHS = 'thousandths'
PIXEL = 'pixel'
COORDINATE_SPACES = (AUTO, NORMALIZED, THOUSANDTHS, PIXEL)


def _token_to_float(token: str) -> float:
    """Convert a numeric token to float, expanding a trailing percent sign."""
    token = token.strip()
    if token.endswith('%'):
        return float(token[:-1]) / 100.0
    return float(token)


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Return the center point of an ``[x1, y1, x2, y2]`` box."""
    x1, y1, x2, y2 = bbox[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def parse_point(prediction: str) -> Optional[Tuple[float, float]]:
    """Parse the predicted click point from a model response.

    The prompt requires the final answer on its own last line after an ``Answer:`` marker,
    so only that line is parsed first.  Restricting the scan to the answer line pins the
    point exactly, so neither preceding reasoning (worked-out pixel arithmetic) nor
    trailing prose (coordinate-system remarks such as "from (0, 0) to (1, 1)") can be
    mistaken for the answer.

    Models that ignore the requested format fall back to scanning the whole response.

    Args:
        prediction (str): Raw or filtered model output.

    Returns:
        Optional[Tuple[float, float]]: The parsed point, or None if nothing matched.
    """
    marker_end = None
    for match in _ANSWER_MARKER_RE.finditer(prediction):
        marker_end = match.end()

    if marker_end is not None:
        # Skip blank lines so that both 'Answer: [x, y]' and 'Answer:\n[x, y]' work,
        # then keep only the answer line itself.
        answer_line = prediction[marker_end:].lstrip().split('\n', 1)[0]
        point = _parse_point_from_text(answer_line, allow_loose_formats=True)
        if point is not None:
            return point

    # Without the marker, only unambiguous point notation is trusted.  Loose scans read
    # layout prose as an answer: 'x=175 to x=935, y=85' describes window bounds and 'the
    # 5th or 6th icon from the left' is an ordinal, yet both yield a confident-looking
    # click point from a truncated reasoning trace.
    return _parse_point_from_text(prediction, allow_loose_formats=False)


def _parse_point_from_text(prediction: str, allow_loose_formats: bool) -> Optional[Tuple[float, float]]:
    """Parse a click point from an arbitrary text span.

    Always accepted, since both unambiguously denote a point: ``<bbox>...</bbox>`` tags
    (reduced to their center) and ``[x, y]`` / ``(x, y)`` pairs.  Accepted only when
    ``allow_loose_formats`` is set: ``x=.., y=..`` pairs and bare numbers.  Each pattern
    matches on its *last* occurrence, which is where the answer sits when a model reasons
    before answering.

    Args:
        prediction (str): Text span to scan.
        allow_loose_formats (bool): Accept formats that are only unambiguous in an
            answer-shaped span.  Never enable this for a whole free-form response.

    Returns:
        Optional[Tuple[float, float]]: The parsed point, or None if nothing matched.
    """
    bbox_tags = re.findall(r'<\s*bbox[^>]*>(.*?)<\s*/\s*bbox\s*>', prediction, flags=re.IGNORECASE | re.DOTALL)
    if bbox_tags:
        tokens = re.findall(_NUM_TOKEN, bbox_tags[-1])
        if len(tokens) >= 4:
            return _bbox_center([_token_to_float(token) for token in tokens[:4]])

    points = re.findall(rf'[\[\(]\s*({_NUM_TOKEN})\s*(?:,|\s)\s*({_NUM_TOKEN})\s*[\]\)]', prediction)
    if points:
        return _token_to_float(points[-1][0]), _token_to_float(points[-1][1])

    if not allow_loose_formats:
        return None

    xy_pairs = re.findall(
        rf'[\'"]?x[\'"]?\s*[:=]\s*({_NUM_TOKEN})[^\d\-]*?[\'"]?y[\'"]?\s*[:=]\s*({_NUM_TOKEN})',
        prediction,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if xy_pairs:
        return _token_to_float(xy_pairs[-1][0]), _token_to_float(xy_pairs[-1][1])

    numbers = re.findall(_NUM_TOKEN, prediction)
    if len(numbers) >= 4 and any(keyword in prediction.lower() for keyword in _BBOX_KEYWORDS):
        return _bbox_center([_token_to_float(number) for number in numbers[-4:]])
    if len(numbers) >= 2:
        return _token_to_float(numbers[-2]), _token_to_float(numbers[-1])
    return None


def normalize_bbox(bbox: Sequence[float], width: int, height: int) -> List[float]:
    """Normalize an absolute ``[x1, y1, x2, y2]`` box to the [0, 1] range."""
    x1, y1, x2, y2 = bbox[:4]
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def to_normalized_point(
    point: Sequence[float],
    image_size: Sequence[int],
    coordinate_space: str = AUTO,
) -> Tuple[float, float]:
    """Map a parsed point onto the normalized [0, 1] coordinate space.

    ``auto`` infers the convention from the magnitude of the coordinates: values inside
    [0, 1] are already normalized, values up to 1000 use the thousandths grid that many
    VLMs emit, and anything larger is read as pixels of the image the model received.
    The 1 < value <= 1000 window is genuinely ambiguous (it could be a pixel coordinate
    on a large screen), so the convention can be pinned explicitly instead.

    Args:
        point (Sequence[float]): The parsed ``(x, y)`` point.
        image_size (Sequence[int]): ``(width, height)`` of the image sent to the model.
        coordinate_space (str): One of ``auto``, ``normalized``, ``thousandths``, ``pixel``.

    Returns:
        Tuple[float, float]: The point expressed in normalized [0, 1] coordinates.
    """
    x, y = point[0], point[1]

    if coordinate_space == AUTO:
        if x <= 1 and y <= 1:
            coordinate_space = NORMALIZED
        elif x <= _THOUSANDTHS_SCALE and y <= _THOUSANDTHS_SCALE:
            coordinate_space = THOUSANDTHS
        else:
            coordinate_space = PIXEL

    if coordinate_space == NORMALIZED:
        return x, y
    if coordinate_space == THOUSANDTHS:
        return x / _THOUSANDTHS_SCALE, y / _THOUSANDTHS_SCALE
    width, height = image_size
    return x / width, y / height


def base64_image_size(data_uri: str) -> Tuple[int, int]:
    """Return the ``(width, height)`` of a base64 data-URI image.

    The adapter needs the size of the image *actually* delivered to the model, which may
    differ from the recorded screenshot size when ``max_image_bytes`` downscales it.
    """
    from PIL import Image

    payload = data_uri.split(',', 1)[-1]
    with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
        return image.size


def point_in_bbox(point: Sequence[float], bbox: Sequence[float]) -> bool:
    """Check whether a point falls inside an ``[x1, y1, x2, y2]`` box (inclusive)."""
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
