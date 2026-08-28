import ast
import math
import random
import re
import string
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUBSET_LIST = ['yaw', 'xy2d', 'depth', 'distance', 'left_right', 'front_behind']
PAIRED_SUBSETS = {'yaw', 'distance', 'left_right', 'front_behind'}
IMAGE_SIZE = (1600, 900)

_RESPONSE_FORMAT = """

Reason carefully and step-by-step under the <think> tag to ensure logical accuracy and robustness, including any relevant error checks.
Finally, provide a concise and definitive response in the <answer> tag. Use the following format:
<think>[Step-by-step reasoning with attention to detail and potential error checks]</think>
<answer>[Final answer]</answer>
"""

YAW_PROMPT = (
    """Task Description:\x20
The primary goal of this task is to identify the direction that the specified object is facing in the given image. The camera in the image is facing {}, and you need to analyze the object's orientation based on this reference.

Question:\x20
Which direction is {} facing in the image?

Options:\x20
- {}
- {}
- {}
- {}"""
    + '\n'
    + _RESPONSE_FORMAT
)

XY2D_PROMPT = (
    """Task Description:
The primary goal of this task is to accurately identify and provide the coordinates of a specified object within a given image. Your task is to analyze the image, locate the object, and return its position in the form of coordinates [x, y].

Question:
Where is {} located in the image?"""
    + _RESPONSE_FORMAT
)

DEPTH_PROMPT = (
    """Task Description:
The primary goal of this task is to estimate the vertical distance of the specified object in the image from the camera, which is positioned at the origin. You need to analyze the image and choose the correct range of distance from the camera based on the visual cues provided.

Question:
How far is the vertical distance of {} in the picture from the camera?

Options:
- {}
- {}
- {}"""
    + _RESPONSE_FORMAT
)

DISTANCE_PROMPT = (
    """Task Description:\x20
The primary goal of this task is to determine which of the two objects is closer to the camera that captured the image below. You need to assess the relative distance between the two objects based on the camera's perspective.

Question:\x20
Which object, {} or {}, is {} to the camera?

Options:
- {}
- {}
- Almost the same"""
    + _RESPONSE_FORMAT
)

LEFT_RIGHT_PROMPT = (
    """Task Description:
The primary goal of this task is to determine the relative left-right positioning of the two objects from the camera's perspective.

Question:
Which is further {}, {} or {}?

Options:
- {}
- {}
- Almost the same"""
    + _RESPONSE_FORMAT
)

FRONT_BEHIND_PROMPT = (
    """Task Description:
The primary goal of this task is to determine the relative front-back positioning of the two objects from the camera's perspective, where the object farther from the camera is considered to be more forward.

Question:
Is {} {} {}?

Options:
- Yes
- No
- Almost the same in terms of front-back position"""
    + _RESPONSE_FORMAT
)

_ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_POINT_PATTERN = re.compile(r'\[\s*([-\d.,\s]+)\s*\]')
_ARTICLE_PATTERN = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)


def extract_tagged_answer(prediction: str) -> str:
    """Extract the first answer enclosed by the official ``<answer>`` tags."""
    match = _ANSWER_PATTERN.search(prediction)
    return match.group(1).strip() if match else ''


def normalize_answer(answer: str) -> str:
    """Apply the official normalization for non-localization answers."""
    answer = answer.strip().lower().translate(_PUNCTUATION_TABLE)
    answer = _ARTICLE_PATTERN.sub('', answer)
    return ' '.join(answer.split())


def option_is_present(option: str, answer: str) -> bool:
    """Return whether an official option occurs as a complete phrase in the answer."""
    return re.search(rf'\b{re.escape(option.strip().lower())}\b', answer.strip().lower()) is not None


def parse_point(answer: str, image_size: Tuple[int, int] = IMAGE_SIZE) -> Optional[Tuple[float, float]]:
    """Parse the official point or bounding-box answer format into pixel coordinates."""
    match = _POINT_PATTERN.search(answer)
    if not match:
        return None

    try:
        coordinates = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) not in (2, 4):
        return None
    if not all(isinstance(value, (int, float)) for value in coordinates):
        return None

    if len(coordinates) == 2:
        x, y = coordinates
    else:
        x1, y1, x2, y2 = coordinates
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2

    width, height = image_size
    x = x * width if x < 1 else x
    y = y * height if y < 1 else y
    if not 0 <= x < width or not 0 <= y < height:
        return None
    return float(x), float(y)


def compute_centerness(point: Tuple[float, float], bbox: Sequence[float]) -> float:
    """Compute the official FCOS-style centerness score inside a target box."""
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    if not xmin <= x <= xmax or not ymin <= y <= ymax:
        return 0.0

    left, right = max(x - xmin, 0.0), max(xmax - x, 0.0)
    top, bottom = max(y - ymin, 0.0), max(ymax - y, 0.0)
    lr_ratio = min(left, right) / max(left, right) if max(left, right) else 1.0
    tb_ratio = min(top, bottom) / max(top, bottom) if max(top, bottom) else 1.0
    return math.sqrt(lr_ratio * tb_ratio)


def _format_range(value_range: Sequence[int]) -> str:
    if len(value_range) != 2:
        raise ValueError(f'Expected a two-value range, got {value_range!r}.')
    start, end = value_range
    return f'Between {start} {"meter" if start <= 1 else "meters"} and {end} {"meter" if end <= 1 else "meters"}'


def _generate_depth_ranges(depth: float, rng: random.Random) -> Tuple[List[int], List[int], List[int]]:
    answer_len = rng.uniform(6, 7)
    answer_range = [max(1, round(depth - answer_len / 2)), round(depth + answer_len / 2)]

    if depth < 7:
        range2_start = answer_range[1] + rng.randint(1, 2)
        range2_end = range2_start + rng.randint(3, 4)
        range3_start = range2_end + rng.randint(1, 2)
        range3_end = range3_start + rng.randint(3, 4)
        range2 = [range2_start, range2_end]
        range3 = [range3_start, range3_end]
    elif depth > 15:
        range2_end = answer_range[0] - rng.randint(1, 2)
        range2_start = range2_end - rng.randint(3, 4)
        range3_end = range2_start - rng.randint(1, 2)
        range3_start = max(1, range3_end - rng.randint(3, 4))
        range2 = [range2_start, range2_end]
        range3 = [range3_start, range3_end]
    else:
        range2_end = answer_range[0] - rng.randint(1, 2)
        range2_start = max(1, range2_end - rng.randint(3, 4))
        range3_start = answer_range[1] + rng.randint(1, 2)
        range3_end = range3_start + rng.randint(3, 4)
        range2 = [range2_start, range2_end]
        range3 = [range3_start, range3_end]

    return answer_range, range2, range3


def _make_unit(
    task: str,
    pair_id: str,
    image_path: str,
    prompts: List[str],
    answers: List[str],
    options: Optional[List[str]] = None,
    bbox: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    return {
        'task': task,
        'pair_id': pair_id,
        'image_path': image_path,
        'prompts': prompts,
        'answers': answers,
        'options': options or [],
        'bbox': list(bbox) if bbox is not None else None,
    }


def build_official_vqa_records(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Reproduce the official seed-42 construction of the six SURDS evaluation tasks."""
    rng = random.Random(42)
    result: Dict[str, List[Dict[str, Any]]] = {subset: [] for subset in SUBSET_LIST}

    for data_index, record in enumerate(records):
        descriptions = record['descs']
        if len(descriptions) <= 1:
            continue
        for first, second in combinations(range(len(descriptions)), 2):
            object1 = f'the {descriptions[first]}'
            object2 = f'the {descriptions[second]}'
            option1, option2 = object1.capitalize(), object2.capitalize()
            pair_suffix = f'{data_index}-{first}-{second}'

            distance1, distance2 = record['distances'][first], record['distances'][second]
            if abs(distance1 - distance2) <= 1:
                closer_answer = farther_answer = 'Almost the same'
            elif distance1 < distance2:
                closer_answer, farther_answer = option1, option2
            else:
                closer_answer, farther_answer = option2, option1
            result['distance'].append(
                _make_unit(
                    'distance',
                    f'distance-{pair_suffix}',
                    record['file_name'],
                    [
                        DISTANCE_PROMPT.format(object1, object2, 'closer', option1, option2),
                        DISTANCE_PROMPT.format(object1, object2, 'farther', option1, option2),
                    ],
                    [closer_answer, farther_answer],
                    options=[option1, option2, 'Almost the same'],
                )
            )

            x1, x2 = record['xy2Ds'][first][0], record['xy2Ds'][second][0]
            if abs(x1 - x2) < 100:
                left_answer = right_answer = 'Almost the same'
            elif x1 < x2:
                left_answer, right_answer = option1, option2
            else:
                left_answer, right_answer = option2, option1
            result['left_right'].append(
                _make_unit(
                    'left_right',
                    f'left-right-{pair_suffix}',
                    record['file_name'],
                    [
                        LEFT_RIGHT_PROMPT.format('left', object1, object2, option1, option2),
                        LEFT_RIGHT_PROMPT.format('right', object1, object2, option1, option2),
                    ],
                    [left_answer, right_answer],
                    options=[option1, option2, 'Almost the same'],
                )
            )

            depth1, depth2 = record['depths'][first], record['depths'][second]
            if abs(depth1 - depth2) < 0.5:
                front_answer = behind_answer = 'Almost the same in terms of front-back position'
            elif depth1 > depth2:
                front_answer, behind_answer = 'Yes', 'No'
            else:
                front_answer, behind_answer = 'No', 'Yes'
            result['front_behind'].append(
                _make_unit(
                    'front_behind',
                    f'front-behind-{pair_suffix}',
                    record['file_name'],
                    [
                        FRONT_BEHIND_PROMPT.format(object1, 'in front of', object2),
                        FRONT_BEHIND_PROMPT.format(object1, 'behind', object2),
                    ],
                    [front_answer, behind_answer],
                    options=['Yes', 'No', 'Almost the same in terms of front-back position'],
                )
            )

    pair_count = len(result['distance'])
    single_object_indices = [index for index, record in enumerate(records) if len(record['descs']) == 1]
    if len(single_object_indices) < pair_count:
        raise ValueError(f'SURDS requires {pair_count} single-object records, found {len(single_object_indices)}.')

    opposite = {
        'North': 'South',
        'South': 'North',
        'East': 'West',
        'West': 'East',
        'Northeast': 'Southwest',
        'Southeast': 'Northwest',
        'Southwest': 'Northeast',
        'Northwest': 'Southeast',
    }
    selected_single_indices = sorted(rng.sample(single_object_indices, pair_count))
    for data_index in selected_single_indices:
        record = records[data_index]
        object_name = f'the {record["descs"][0]}'
        yaw_answer = record['yaw_descs'][0]
        diagonal_options = rng.sample(['Northeast', 'Southeast', 'Northwest', 'Southwest'], k=4)
        cardinal_options = rng.sample(['East', 'South', 'West', 'North'], k=4)
        if yaw_answer in diagonal_options:
            yaw_options = diagonal_options
        elif yaw_answer in cardinal_options:
            yaw_options = cardinal_options
        else:
            raise ValueError(f'Unsupported yaw label {yaw_answer!r}.')

        result['yaw'].append(
            _make_unit(
                'yaw',
                f'yaw-{data_index}',
                record['file_name'],
                [
                    YAW_PROMPT.format('North', object_name, *yaw_options),
                    YAW_PROMPT.format('South', object_name, *yaw_options),
                ],
                [yaw_answer, opposite[yaw_answer]],
                options=yaw_options,
                bbox=record['bboxes2D'][0],
            )
        )
        result['xy2d'].append(
            _make_unit(
                'xy2d',
                f'xy2d-{data_index}',
                record['file_name'],
                [XY2D_PROMPT.format(object_name)],
                [str(record['xy2Ds'][0])],
                bbox=record['bboxes2D'][0],
            )
        )

        answer_range, distractor1, distractor2 = _generate_depth_ranges(record['depths'][0], rng)
        options = rng.sample([_format_range(answer_range), _format_range(distractor1), _format_range(distractor2)], k=3)
        result['depth'].append(
            _make_unit(
                'depth',
                f'depth-{data_index}',
                record['file_name'],
                [DEPTH_PROMPT.format(object_name, *options)],
                [_format_range(answer_range)],
                options=options,
                bbox=record['bboxes2D'][0],
            )
        )

    return result
