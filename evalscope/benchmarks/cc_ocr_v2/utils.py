# flake8: noqa: E501
"""Per-sample scoring for CC-OCR V2, ported from the official evaluation scripts.

Reference: https://github.com/eioss/CC-OCR-V2 (``src/evaluate_*.py``). Each track keeps its
own metric, and every function below scores a single sample so the framework's ``mean``
aggregation reproduces the official per-dataset averages.
"""

import ast
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evalscope.utils.logger import get_logger

logger = get_logger()

_CODE_FENCE_RE = re.compile(r'^```(?:json)?\s*(.*?)\s*```$', re.DOTALL | re.IGNORECASE)
_TABLE_BLOCK_RE = re.compile(r'<table\b[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL)
_CUSTOM_SEP_RE = re.compile(r'[/／|｜]+')
_BBOX_RE = re.compile(r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]')

# LaTeX preamble commands dropped from document-parsing predictions before scoring.
_LATEX_PREAMBLE_PATTERNS = [
    r'\\documentclass\{.*?\}',
    r'\\usepackage\[.*?\]\{.*?\}',
    r'\\usepackage\{.*?\}',
    r'\\geometry\{.*?\}',
    r'\\begin\{document\}',
    r'\\end\{document\}',
    r'\\noindent',
]

_LATEX_META_TAIL = re.compile(
    r'(?is)\n\s*(?:'
    r'#{1,6}\s*(?:📝\s*)?(?:Notes(?:\s+on\s+Transcription)?\s*:?[^\n]*)'
    r'|📌\s*Notes[^\n]*'
    r')(?:\n.*)*\Z'
)
_LATEX_LEADING_FLUFF = re.compile(
    r'(?is)^(?:'
    r'Here (?:is|’s|\'s|are) (?:the )?(?:LaTeX|latex) [^\n]+(?:\n|$)'
    r'|Note that [^\n]+(?:\n|$)'
    r')+'
)
_HTML_META_TAIL = re.compile(
    r'(?is)\n\s*(?:'
    r'#{1,6}\s*(?:📝\s*)?Notes[^\n]*'
    r'|📌\s*Notes[^\n]*'
    r'|✅\s*\*{0,2}Note\*{0,2}\s*:'
    r'|>{0,1}\s*\*{0,2}\s*Note\*{0,2}\s*:'
    r')(?:\n.*)*\Z'
)

# Weight of the table TEDS term in the mixed text+table score used by info-board parsing.
_CUSTOM_TABLE_WEIGHT = 0.9

# ##########################
# SHARED TEXT HELPERS
# ##########################


def strip_code_fence(text: str) -> str:
    """Return the body of a fully fenced code block, or the text unchanged.

    The pattern is copied from the official scorers (``evaluate_recognition.py``,
    ``evaluate_vqa.py``, ``evaluate_grounding.py``), which accept a bare or ``json`` fence only.
    An ``html`` or ``latex`` fence still loses its backticks, but the language tag itself is not
    consumed and so leaks into the scored text and costs the prediction points. The parsing track
    therefore unwraps those two languages explicitly instead of relying on this helper.
    """
    text = text.strip()
    match = _CODE_FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def has_cjk(text: str) -> bool:
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def convert_to_halfwidth(text: str) -> str:
    table = str.maketrans(
        '！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～',
        '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
    )
    return text.translate(table)


def text_normalize_and_tokenize(
    text: str,
    is_keep_blank: bool = True,
    is_lower: bool = True,
    is_alphanum_only: bool = False,
) -> List[str]:
    """Tokenize OCR text into words (``is_keep_blank``) or characters."""
    text = text.replace('\t', ' ').replace('\n', ' ').replace('###', '').replace('***', '')
    text = re.sub(r'\s+', ' ', text)
    if not is_keep_blank:
        text = text.replace(' ', '')
    tokens = text.split(' ') if is_keep_blank else list(text)
    if is_lower:
        tokens = [token.lower() for token in tokens]
    if is_alphanum_only:
        tokens = [re.sub('[^A-Za-z0-9]+', '', token) for token in tokens]
    return [token for token in tokens if token]


def token_multiset_f1(gt_tokens: Sequence[str], pred_tokens: Sequence[str]) -> float:
    """F1 over token multisets. Single-sample micro and macro F1 coincide, so one
    implementation serves both the recognition track and the info-board text term."""
    pred_counter = Counter(pred_tokens)
    right_num = sum(min(count, pred_counter.get(token, 0)) for token, count in Counter(gt_tokens).items())
    recall = right_num / (len(gt_tokens) + 1e-9)
    precision = right_num / (len(pred_tokens) + 1e-9)
    return 2 * recall * precision / (recall + precision + 1e-9)


def edit_similarity(pred: str, gt: str) -> float:
    """``1 - normalized Levenshtein distance``; two empty strings count as identical."""
    from evalscope.metrics.utils.functions import levenshtein_distance

    length = max(len(pred), len(gt))
    if length == 0:
        return 1.0
    return 1 - levenshtein_distance(pred, gt) / length


# ##########################
# RECOGNITION
# ##########################


def score_recognition(prediction: str, reference: str, scenario: str) -> float:
    """Token-level F1, scored per character when the reference contains a CJK ideograph.

    This mirrors the CC-OCR V2 entry point ``evaluate_recognition.py::main`` (``--mode auto``),
    which switches to character level from ``_has_cjk`` on the reference text alone. Arabic and
    Korean therefore stay word-level here: the ``dataset_name in ["Arabic", "Japanese",
    "Korean"]`` rule belongs to the CC-OCR v1 ``OcrEvaluator`` class, which V2 no longer calls.
    Word-level multi-scene data additionally keeps alphanumeric characters only, matching the
    official ``_path_indicates_multi_scene``.
    """
    gt_text = reference.strip()
    pred_text = strip_code_fence(prediction).strip()

    is_word_level = not has_cjk(gt_text)
    is_alphanum_only = is_word_level and scenario.startswith('multi_scene_ocr')

    gt_tokens = text_normalize_and_tokenize(gt_text, is_word_level, True, is_alphanum_only)
    pred_tokens = text_normalize_and_tokenize(pred_text, is_word_level, True, is_alphanum_only)
    return token_multiset_f1(gt_tokens, pred_tokens)


# ##########################
# KEY INFORMATION EXTRACTION
# ##########################


def _fullwidth_to_halfwidth(text: str) -> str:
    result = ''
    for char in text:
        code_point = ord(char)
        if code_point == 0x3000:
            code_point = 0x0020
        elif code_point == 0xFFE5:
            code_point = 0x00A5
        elif code_point == 0x2014:
            code_point = 0x002D
        elif code_point == 0x2103:
            result += chr(0x00B0) + 'C'
            continue
        elif 0xFF01 <= code_point <= 0xFF5E:
            code_point -= 0xFEE0
        result += chr(code_point)
    result = result.replace('、', ',')
    result = result.replace('-', '')
    result = result.replace('–', '')
    result = result.replace('’', "'")
    return result.rstrip('。.')


def _remove_unnecessary_spaces(text: str) -> str:
    if '```json' in text:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif '```' in text:
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return re.sub(r'\s+', '', text)


def _normalize_kie_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_kie_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_kie_value(item) for item in value]
    if isinstance(value, str):
        return _remove_unnecessary_spaces(_fullwidth_to_halfwidth(value))
    return value


def _normalize_kie_dict(data: Any) -> Any:
    """Sort keys and coerce leaves to lists of stripped strings (donut-style normalization)."""
    if isinstance(data, dict):
        new_data = {}
        for key in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = _normalize_kie_dict(data[key])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new_data[key] = value
        return new_data
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return [item for item in (_normalize_kie_dict(entry) for entry in data) if item]
        return [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
    return [str(data).strip()]


def _flatten_kie_fields(data: Any) -> List[Tuple[str, Any]]:
    """Flatten a nested dict into ``(dotted_key, leaf_value)`` pairs."""
    flat: List[Tuple[str, Any]] = []

    def _walk(value: Any, key: str = '') -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                _walk(child_value, f'{key}.{child_key}' if key else child_key)
        elif isinstance(value, list):
            for item in value:
                _walk(item, key)
        else:
            flat.append((key, value))

    _walk(data)
    return flat


def _parse_kie_json(text: str) -> Any:
    content = text.strip()
    if '```json' in content:
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)
    elif '```' in content:
        match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def score_kie(prediction: str, reference: str) -> float:
    """Field-level F1 between the predicted and reference JSON objects."""
    pred = _parse_kie_json(prediction)
    gt = _parse_kie_json(reference)
    if gt is None:
        logger.warning('CC-OCR V2 extraction reference is not valid JSON; scoring the sample as 0.')
        return 0.0

    pred_fields = _flatten_kie_fields(_normalize_kie_dict(_normalize_kie_value(pred if pred is not None else {})))
    gt_fields = _flatten_kie_fields(_normalize_kie_dict(_normalize_kie_value(gt)))

    true_positive, mismatch = 0, 0
    for field in pred_fields:
        if field in gt_fields:
            true_positive += 1
            gt_fields.remove(field)
        else:
            mismatch += 1
    mismatch += len(gt_fields)
    return true_positive / (true_positive + mismatch / 2 + 1e-6)


# ##########################
# DOCUMENT QUESTION ANSWERING
# ##########################


def _parse_vqa_reference(reference: str) -> Any:
    raw = reference.strip()
    if raw.startswith('['):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                data = None
        if isinstance(data, list) and data:
            return data
    return raw


def _anls(predict: str, answer: str) -> float:
    from evalscope.metrics.utils.functions import levenshtein_distance

    length = max(len(predict), len(answer))
    if length == 0:
        return 0.0
    value = 1 - levenshtein_distance(predict, answer) / length
    return value if value >= 0.5 else 0.0


def _score_vqa_answer(predict: str, answer: str, is_chinese: bool, anls_fallback: bool) -> float:
    """Substring match for short answers, ANLS for long ones (official CC-OCR V2 rule).

    ``anls_fallback`` mirrors ``cn_vqa_evaluation``, which -- unlike every other branch of the
    official scorers -- also falls back to ANLS when a short Chinese answer is not contained
    in the prediction.
    """
    if is_chinese:
        answer = answer.lower().strip().replace('\n', ' ').replace(' ', '')
        predict = predict.lower().strip().replace('\n', ' ').replace(' ', '')
        is_short = len(answer.split(',')) < 4
    else:
        answer = answer.lower().strip().replace('\n', ' ')
        predict = predict.lower().strip().replace('\n', ' ')
        is_short = len(answer.split()) < 5

    if not is_short:
        return _anls(predict, answer)
    if answer in predict:
        return 1.0
    return _anls(predict, answer) if anls_fallback else 0.0


def score_vqa(prediction: str, reference: str) -> float:
    predict = strip_code_fence(prediction).strip()
    if not predict:
        return 0.0

    answers = _parse_vqa_reference(reference)
    is_list = isinstance(answers, list)
    answer_list = [str(item) for item in answers] if is_list else [str(answers)]
    is_chinese = has_cjk(' '.join(answer_list))
    return max(_score_vqa_answer(predict, answer, is_chinese, is_chinese and not is_list) for answer in answer_list)


# ##########################
# GROUNDING
# ##########################


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a[:4])
    bx1, by1, bx2, by2 = (float(value) for value in box_b[:4])
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def parse_point_box(text: str) -> Optional[List[float]]:
    """Extract the first ``(x1, y1, x2, y2)`` tuple from free-form text."""
    match = _BBOX_RE.search(strip_code_fence(text))
    if not match:
        return None
    return [float(match.group(index)) for index in range(1, 5)]


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def normalize_bbox_xyxy(bbox: Any) -> Optional[List[float]]:
    """Coerce the bbox notations emitted by different model families into ``[x1, y1, x2, y2]``."""
    if isinstance(bbox, (list, tuple)):
        if len(bbox) < 4:
            return None
        values = [_to_float(bbox[index]) for index in range(4)]
        return None if any(value is None for value in values) else values
    if not isinstance(bbox, dict):
        return None

    lowered = {str(key).lower().strip(): value for key, value in bbox.items() if key is not None}

    def get(*names: str) -> Optional[float]:
        for name in names:
            if name in lowered:
                value = _to_float(lowered[name])
                if value is not None:
                    return value
        return None

    for keys in (('x1', 'y1', 'x2', 'y2'), ('x1', 'y', 'x2', 'y2'), ('xmin', 'ymin', 'xmax', 'ymax'),
                 ('left', 'top', 'right', 'bottom'), ('x0', 'y0', 'x1', 'y1')):
        corners = [get(key) for key in keys]
        if all(corner is not None for corner in corners):
            return corners

    x, y = get('x'), get('y')
    width, height = get('width', 'w'), get('height', 'h')
    if None not in (x, y, width, height):
        return [x, y, x + width, y + height]
    return None


def parse_object_list(text: str) -> Optional[List[List[float]]]:
    """Parse a JSON array of detections into a list of ``[x1, y1, x2, y2]`` boxes."""
    content = strip_code_fence(text)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(data, list):
        return None

    boxes = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = item.get('label', item.get('category_name'))
        bbox = item.get('bbox_2d') or item.get('bbox') or item.get('box_2d')
        if label is None or not bbox:
            continue
        xyxy = normalize_bbox_xyxy(bbox)
        if xyxy is not None:
            boxes.append(xyxy)
    return boxes


def _image_size(image_path: str) -> Optional[Tuple[int, int]]:
    if not image_path or not os.path.exists(image_path):
        return None
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _to_pixel_boxes(boxes: List[List[float]], size: Optional[Tuple[int, int]]) -> List[List[float]]:
    """Rescale predicted boxes to pixels. Values within ``[0, 1]`` are treated as unit
    normalized, anything larger as the 0-1000 grid requested by the prompt."""
    if not size:
        return boxes
    width, height = size
    scale = 1 if max(value for box in boxes for value in box[:4]) <= 1.0 else 1000
    return [[box[0] * width / scale, box[1] * height / scale, box[2] * width / scale, box[3] * height / scale]
            for box in boxes]


def score_text_grounding(prediction: str, reference: str, image_path: str) -> float:
    try:
        gt_box = ast.literal_eval(reference.strip())
    except (ValueError, SyntaxError):
        gt_box = None
    if not isinstance(gt_box, (list, tuple)) or len(gt_box) < 4:
        logger.warning('CC-OCR V2 text grounding reference is not a bounding box; scoring the sample as 0.')
        return 0.0

    pred_box = parse_point_box(prediction)
    if pred_box is None:
        return 0.0
    return iou_xyxy(_to_pixel_boxes([pred_box], _image_size(image_path))[0], gt_box)


def score_object_grounding(prediction: str, reference: str, image_path: str) -> float:
    """Mean IoU over ground-truth boxes after optimal one-to-one matching."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    gt_boxes = parse_object_list(reference)
    if not gt_boxes:
        logger.warning('CC-OCR V2 object grounding reference has no boxes; scoring the sample as 0.')
        return 0.0

    pred_boxes = parse_object_list(prediction)
    if not pred_boxes:
        return 0.0

    pred_boxes = _to_pixel_boxes(pred_boxes, _image_size(image_path))
    iou_matrix = np.array([[iou_xyxy(gt, pred) for pred in pred_boxes] for gt in gt_boxes])
    rows, cols = linear_sum_assignment(-iou_matrix)
    return float(iou_matrix[rows, cols].sum() / len(gt_boxes))


# ##########################
# DOCUMENT PARSING
# ##########################


def parsing_op(scenario: str) -> str:
    """Map a parsing scenario directory name to its scoring mode."""
    name = scenario.lower()
    for keyword in ('formula', 'molecular', 'custom', 'table'):
        if keyword in name:
            return keyword
    return 'doc'


def extract_and_clean_tables(text: str) -> str:
    """Concatenate all ``<table>`` blocks with whitespace collapsed, as TEDS expects."""
    if '</table>' not in text:
        text += '</table>'

    tables = []
    for table in re.findall(r'<table.*?>.*?</table>', text, re.DOTALL):
        table = re.sub(r'<table.*?>', '<table>', table)
        table = re.sub(r'>\s+<', '><', table)
        table = re.sub(
            r'>(.*?)<', lambda m: '>' + m.group(1).replace('\n', '').replace(' ', '') + '<', table, flags=re.DOTALL
        )
        tables.append(table.replace('\n', '').strip())
    return ''.join(tables)


def _strip_latex_chatter(text: str) -> str:
    text = _LATEX_META_TAIL.sub('', (text or '').strip())
    if not text:
        return text
    starts = [
        match.start()
        for match in (re.search(r'(?m)^\\documentclass\b', text), re.search(r'(?m)^\\begin\s*\{', text))
        if match is not None
    ]
    if starts:
        text = text[min(starts):]
    else:
        text = _LATEX_LEADING_FLUFF.sub('', text)
    return text.strip()


def _strip_html_chatter(text: str) -> str:
    text = _HTML_META_TAIL.sub('', (text or '').strip())
    if not text:
        return text
    match = re.search(r'(?i)<\s*(?:!DOCTYPE\b|html\b|body\b|table\b|div\b|h[1-6]\b|p\b|main\b|section\b)', text)
    if match is not None:
        text = text[match.start():]
    return text.lstrip()


def _unwrap_fenced_block(text: str, tag: str) -> str:
    """Pull out the body of a ```<tag> block even when the closing fence is missing."""
    body = text.strip()
    fence = f'```{tag}'
    if fence in body:
        rest = body.split(fence, 1)[1]
        body = rest.split('```', 1)[0] if '```' in rest else rest
    return body.strip()


def _unwrap_fenced_html(prediction: str) -> str:
    return _strip_html_chatter(_unwrap_fenced_block(prediction, 'html'))


def _teds_score(gt_html: str, pred_html: str) -> float:
    # Reuse the OCRBench-v2 TEDS implementation instead of shipping a second copy.
    from evalscope.benchmarks.ocr_bench.ocr_bench_v2.TEDS_metric import TEDS

    try:
        return float(TEDS(structure_only=False, n_jobs=1).evaluate(pred_html, gt_html))
    except Exception as error:
        # Truncated or malformed predictions can produce HTML that the tree builder rejects
        # (e.g. a non-numeric colspan); such a prediction simply earns no credit.
        logger.warning(f'CC-OCR V2 TEDS computation failed, scoring the table as 0: {error}')
        return 0.0


def _wrap_html(fragment: str) -> str:
    return f'<html><body>{fragment}</body></html>'


def _score_parsing_doc(prediction: str, reference: str) -> float:
    for pattern in _LATEX_PREAMBLE_PATTERNS:
        prediction = re.sub(pattern, '', prediction)
    prediction = _strip_latex_chatter(_unwrap_fenced_block(prediction, 'latex'))
    return edit_similarity(prediction.replace(' ', '').replace('\n', ''), reference.replace(' ', '').replace('\n', ''))


def _score_parsing_table(prediction: str, reference: str) -> float:
    pred = convert_to_halfwidth(extract_and_clean_tables(_unwrap_fenced_html(prediction)))
    gt = convert_to_halfwidth(extract_and_clean_tables(reference))
    return _teds_score(_wrap_html(gt), _wrap_html(pred))


def _score_parsing_formula(prediction: str, reference: str) -> float:
    pred = prediction.strip().replace('\n', ' ').replace('```latex',
                                                         '').replace('```', '').replace('\t', ' ').replace(' ', '')
    return edit_similarity(pred, reference.replace(' ', ''))


def _score_parsing_molecular(prediction: str, reference: str) -> float:
    pred = prediction.strip().replace('\n', '').replace(' ', '').replace('<smiles>', '').replace('</smiles>', '')
    return edit_similarity(pred, reference.replace(' ', ''))


def _html_to_plain_text(fragment: str) -> str:
    if not fragment.strip():
        return ''
    from lxml import html as lxml_html

    parser = lxml_html.HTMLParser(encoding='utf-8', remove_comments=True)
    root = lxml_html.fromstring(f'<div>{fragment}</div>', parser=parser)
    return (root.text_content() or '').strip()


def _custom_text_tokens(mixed: str) -> List[str]:
    plain = _html_to_plain_text(_TABLE_BLOCK_RE.sub('', mixed))
    plain = re.sub(r'\s+', ' ', _CUSTOM_SEP_RE.sub(' ', convert_to_halfwidth(plain))).strip()
    return text_normalize_and_tokenize(plain, is_keep_blank=False, is_lower=False)


def _custom_table_teds(gt_raw: str, pred_inner: str) -> float:
    """Average TEDS over tables paired by document order."""
    from lxml import etree
    from lxml import html as lxml_html

    gt_flat = extract_and_clean_tables(gt_raw).strip()
    pred_flat = extract_and_clean_tables(pred_inner).strip()
    if not gt_flat and not pred_flat:
        return 1.0
    if not gt_flat or not pred_flat:
        return 0.0

    parser = lxml_html.HTMLParser(encoding='utf-8', remove_comments=True)
    gt_tables = lxml_html.fromstring(_wrap_html(convert_to_halfwidth(gt_flat)), parser=parser).xpath('body/table')
    pred_tables = lxml_html.fromstring(_wrap_html(convert_to_halfwidth(pred_flat)), parser=parser).xpath('body/table')
    if not gt_tables or not pred_tables:
        return 0.0

    scores = []
    for index in range(max(len(gt_tables), len(pred_tables))):
        if index >= len(gt_tables) or index >= len(pred_tables):
            scores.append(0.0)
            continue
        gt_html = _wrap_html(etree.tostring(gt_tables[index], encoding='unicode', method='html'))
        pred_html = _wrap_html(etree.tostring(pred_tables[index], encoding='unicode', method='html'))
        scores.append(_teds_score(gt_html, pred_html))
    return sum(scores) / len(scores)


def _score_parsing_custom(prediction: str, reference: str) -> float:
    pred_inner = _unwrap_fenced_html(prediction)
    gt_tokens = _custom_text_tokens(reference)
    pred_tokens = _custom_text_tokens(pred_inner)
    text_score = 1.0 if not gt_tokens and not pred_tokens else token_multiset_f1(gt_tokens, pred_tokens)
    table_score = _custom_table_teds(reference, pred_inner)
    return (1.0 - _CUSTOM_TABLE_WEIGHT) * text_score + _CUSTOM_TABLE_WEIGHT * table_score


def score_parsing(prediction: str, reference: str, scenario: str) -> float:
    op = parsing_op(scenario)
    if op == 'formula':
        return _score_parsing_formula(prediction, reference)
    if op == 'molecular':
        return _score_parsing_molecular(prediction, reference)
    if op == 'custom':
        return _score_parsing_custom(prediction, reference)
    if op == 'table':
        return _score_parsing_table(prediction, reference)
    return _score_parsing_doc(prediction, reference)


# ##########################
# DISPATCH
# ##########################


def score_sample(prediction: str, reference: str, metadata: Dict[str, Any]) -> float:
    """Score one sample with the metric of its track."""
    task = metadata.get('task', '')
    scenario = metadata.get('scenario', '')

    if task == 'recognition':
        return score_recognition(prediction, reference, scenario)
    if task == 'extraction':
        return score_kie(prediction, reference)
    if task == 'qa':
        return score_vqa(prediction, reference)
    if task == 'parsing':
        return score_parsing(prediction, reference, scenario)
    if task == 'grounding':
        image_path = (metadata.get('image_paths') or [''])[0]
        if metadata.get('sub_task') == 'object_grounding':
            return score_object_grounding(prediction, reference, image_path)
        return score_text_grounding(prediction, reference, image_path)
    raise ValueError(f'Unknown CC-OCR V2 task: {task!r}')
