import argparse
import html
import json
import math
import numpy as np
import os
import pandas as pd
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from shapely.geometry import Polygon
from typing import Dict, List, Optional, Tuple


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


# =========================================================
# Part 1. Layout / DIoU Evaluation
# =========================================================

Point = Tuple[float, float]
Quad = List[Point]

_PAIR_RE = re.compile(
    r'<\|object_ref_start\|>(?P<text>.*?)<\|object_ref_end\|>'
    r'<\|quad_start\|>\s*'
    r'\((?P<x1>-?\d+(?:\.\d+)?),(?P<y1>-?\d+(?:\.\d+)?)\)\s*,\s*'
    r'\((?P<x2>-?\d+(?:\.\d+)?),(?P<y2>-?\d+(?:\.\d+)?)\)\s*,\s*'
    r'\((?P<x3>-?\d+(?:\.\d+)?),(?P<y3>-?\d+(?:\.\d+)?)\)\s*,\s*'
    r'\((?P<x4>-?\d+(?:\.\d+)?),(?P<y4>-?\d+(?:\.\d+)?)\)\s*'
    r'<\|quad_end\|>',
    flags=re.DOTALL,
)


def normalize_pair_text(s: str, do_normalize: bool = True) -> str:
    s = s.strip()
    if do_normalize:
        s = re.sub(r'\s+', ' ', s)
    return s


def parse_all_pairs(s: str, normalize: bool = True) -> List[Tuple[str, Quad]]:
    """
    Parse all (text, quad) pairs from the string without deduplication.
    """
    out: List[Tuple[str, Quad]] = []
    if not s:
        return out

    for m in _PAIR_RE.finditer(s):
        text = normalize_pair_text(m.group('text'), do_normalize=normalize)
        quad: Quad = [
            (float(m.group('x1')), float(m.group('y1'))),
            (float(m.group('x2')), float(m.group('y2'))),
            (float(m.group('x3')), float(m.group('y3'))),
            (float(m.group('x4')), float(m.group('y4'))),
        ]
        out.append((text, quad))
    return out


def reorder_quad_points(quad: Quad) -> Quad:
    """
    Sort the four points by polar angle around the center to reduce self-intersections.
    """
    cx = sum(x for x, y in quad) / 4.0
    cy = sum(y for x, y in quad) / 4.0

    points_with_angle = []
    for x, y in quad:
        angle = math.atan2(y - cy, x - cx)
        points_with_angle.append((x, y, angle))

    points_with_angle.sort(key=lambda p: p[2])
    ordered = [(x, y) for x, y, _ in points_with_angle]
    return ordered


def make_valid_polygon(quad: Quad) -> Polygon:
    """
    Convert a quadrilateral into a polygon and repair invalid geometry when possible.
    """
    quad = reorder_quad_points(quad)
    poly = Polygon(quad)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def polygon_diou_score(
    poly1: Polygon,
    poly2: Polygon,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
    eps: float = 1e-9,
) -> float:
    """
    DIoU score aligned with the reward function:
        diou = iou - center_dist_sq / diag_length_sq
        score = (diou + 1) / 2
        final score is clipped to [min_reward, max_reward]
    """
    if poly1.is_empty or poly2.is_empty:
        return min_reward

    area1 = poly1.area
    area2 = poly2.area
    if area1 <= eps or area2 <= eps:
        return min_reward

    inter_area = poly1.intersection(poly2).area
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > eps else 0.0
    iou = max(0.0, min(1.0, iou))

    c1 = poly1.centroid
    c2 = poly2.centroid
    center_dist_sq = (c1.x - c2.x)**2 + (c1.y - c2.y)**2

    minx1, miny1, maxx1, maxy1 = poly1.bounds
    minx2, miny2, maxx2, maxy2 = poly2.bounds

    x_min = min(minx1, minx2)
    y_min = min(miny1, miny2)
    x_max = max(maxx1, maxx2)
    y_max = max(maxy1, maxy2)

    diag_length_sq = (x_max - x_min)**2 + (y_max - y_min)**2

    if diag_length_sq < eps:
        diou = iou
    else:
        diou = iou - center_dist_sq / diag_length_sq

    score = (diou + 1.0) / 2.0
    score = max(min_reward, min(max_reward, score))
    return score


def best_match_average(
    source_polys: List[Polygon],
    target_polys: List[Polygon],
    min_reward: float = 0.0,
    max_reward: float = 1.0,
    eps: float = 1e-9,
) -> float:
    """
    For each source polygon, find the best target match and average the scores.
    """
    if not source_polys or not target_polys:
        return min_reward

    scores = []
    for src_poly in source_polys:
        best_score = min_reward
        for tgt_poly in target_polys:
            score = polygon_diou_score(
                src_poly,
                tgt_poly,
                min_reward=min_reward,
                max_reward=max_reward,
                eps=eps,
            )
            if score > best_score:
                best_score = score
        scores.append(best_score)

    return float(np.mean(scores)) if scores else min_reward


def evaluate_layout_sample(
    completion: str,
    solution: str,
    normalize_text_flag: bool = True,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
    blank_reward: Optional[float] = 0.3,
    pred_to_gt_weight: float = 0.5,
    eps: float = 1e-9,
) -> Dict[str, float]:
    """
    Evaluate layout for a single sample, aligned with TextQuadPolygonDIoUReward.
    """
    gt_to_pred_weight = 1.0 - pred_to_gt_weight
    blank_reward = max_reward if blank_reward is None else blank_reward

    pred_items = parse_all_pairs(completion, normalize=normalize_text_flag)
    gt_items = parse_all_pairs(solution, normalize=normalize_text_flag)

    pred_quads = [quad for _, quad in pred_items]
    gt_quads = [quad for _, quad in gt_items]

    pred_polys = [make_valid_polygon(q) for q in pred_quads]
    gt_polys = [make_valid_polygon(q) for q in gt_quads]

    pred_polys = [p for p in pred_polys if not p.is_empty and p.area > eps]
    gt_polys = [p for p in gt_polys if not p.is_empty and p.area > eps]

    # Blank page: ground truth is empty.
    if not gt_polys:
        final_score = blank_reward if not pred_polys else min_reward
        return {'diou_score': float(final_score)}

    # Normal page: ground truth exists but there is no prediction.
    if not pred_polys:
        return {'diou_score': float(min_reward)}

    pred_to_gt_score = best_match_average(
        pred_polys,
        gt_polys,
        min_reward=min_reward,
        max_reward=max_reward,
        eps=eps,
    )
    gt_to_pred_score = best_match_average(
        gt_polys,
        pred_polys,
        min_reward=min_reward,
        max_reward=max_reward,
        eps=eps,
    )

    final_score = (pred_to_gt_weight * pred_to_gt_score + gt_to_pred_weight * gt_to_pred_score)
    final_score = max(min_reward, min(max_reward, final_score))

    return {'diou_score': float(final_score)}


# =========================================================
# Part 2. Text Evaluation
# =========================================================

QUAD_BLOCK_RE = re.compile(r'<\|quad_start\|>.*?<\|quad_end\|>', flags=re.DOTALL)
OBJECT_START_RE = re.compile(r'<\|object_ref_start\|>')
OBJECT_END_RE = re.compile(r'<\|object_ref_end\|>')
ANY_SPECIAL_TOKEN_RE = re.compile(r'<\|.*?\|>')
MULTISPACE_RE = re.compile(r'\s+')

TABLE_TAG_HINT_RE = re.compile(
    r'<\s*(table|tr|td|th|thead|tbody|tfoot|br)\b|<\s*/\s*(table|tr|td|th|thead|tbody|tfoot|br)\s*>',
    flags=re.IGNORECASE,
)
GENERIC_HTML_TAG_RE = re.compile(r'<[^<>]+>')


def contains_table_markup(text: str) -> bool:
    if not text:
        return False
    return bool(TABLE_TAG_HINT_RE.search(text))


def strip_special_and_coords(text: str) -> str:
    if not text:
        return ''

    text = QUAD_BLOCK_RE.sub(' ', text)
    text = OBJECT_START_RE.sub('', text)
    text = OBJECT_END_RE.sub('', text)
    text = ANY_SPECIAL_TOKEN_RE.sub(' ', text)
    return text


def html_to_text(text: str) -> str:
    if not text:
        return ''

    text = html.unescape(text)

    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?\s*(table|thead|tbody|tfoot)\b[^<>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?\s*(tr)\b[^<>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?\s*(td|th)\b[^<>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?\s*(p|div|span)\b[^<>]*>', ' ', text, flags=re.IGNORECASE)

    text = GENERIC_HTML_TAG_RE.sub(' ', text)
    return text


def normalize_unicode(text: str) -> str:
    if not text:
        return ''
    return unicodedata.normalize('NFKC', text)


def normalize_punctuation(text: str) -> str:
    if not text:
        return ''

    mapping = {
        '，': ',',
        '、': ',',
        '；': ';',
        '：': ':',
        '。': '.',
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '—': '-',
        '–': '-',
        '－': '-',
        '·': '.',
        '•': '.',
        '…': '...',
        '\u00a0': ' ',
        '\t': ' ',
        '\n': ' ',
        '\r': ' ',
    }

    out = []
    for ch in text:
        out.append(mapping.get(ch, ch))
    return ''.join(out)


def basic_normalize(text: str, lowercase: bool = True) -> str:
    text = strip_special_and_coords(text)
    text = html_to_text(text)
    text = normalize_unicode(text)
    text = normalize_punctuation(text)

    if lowercase:
        text = text.lower()

    text = MULTISPACE_RE.sub(' ', text).strip()
    return text


def compact_text(text: str) -> str:
    return re.sub(r'\s+', '', text)


def loose_text(text: str) -> str:
    text = compact_text(text)
    text = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]', '', text)
    return text


def table_aware_text(raw_text: str) -> str:
    text = strip_special_and_coords(raw_text)
    text = html_to_text(text)
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    text = text.lower()
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]', '', text)
    return text


def edit_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0

    n, m = len(a), len(b)
    dp = [0] * (m + 1)

    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def lcs_f1(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    lcs = lcs_length(a, b)
    p = safe_div(lcs, len(a))
    r = safe_div(lcs, len(b))
    return safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0


def char_f1(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    ca = Counter(a)
    cb = Counter(b)
    overlap = sum((ca & cb).values())

    p = safe_div(overlap, sum(ca.values()))
    r = safe_div(overlap, sum(cb.values()))
    return safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0


def repetition_penalty(text: str) -> float:
    if not text:
        return 1.0
    if len(text) < 20:
        return 1.0

    uniq_ratio = len(set(text)) / max(len(text), 1)
    if uniq_ratio < 0.08:
        return 0.92
    if uniq_ratio < 0.12:
        return 0.96
    return 1.0


def length_penalty(pred: str, gt: str) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.9

    ratio = min(len(pred), len(gt)) / max(len(pred), len(gt))
    if ratio < 0.2:
        return 0.90
    if ratio < 0.4:
        return 0.95
    return 1.0


def table_compat_bonus(raw_pred: str, raw_gt: str, content_sim: float) -> float:
    pred_has_table = contains_table_markup(raw_pred)
    gt_has_table = contains_table_markup(raw_gt)

    if pred_has_table != gt_has_table:
        if content_sim >= 0.98:
            return 0.10
        if content_sim >= 0.95:
            return 0.08
        if content_sim >= 0.92:
            return 0.06
        if content_sim >= 0.88:
            return 0.04
        if content_sim >= 0.82:
            return 0.02
    return 0.0


def evaluate_text_sample(completion: str, solution: str) -> Dict[str, float]:
    raw_pred = completion or ''
    raw_gt = solution or ''

    pred_v1 = basic_normalize(raw_pred, lowercase=True)
    gt_v1 = basic_normalize(raw_gt, lowercase=True)

    pred_v2 = compact_text(pred_v1)
    gt_v2 = compact_text(gt_v1)

    pred_v3 = loose_text(pred_v1)
    gt_v3 = loose_text(gt_v1)

    pred_v4 = table_aware_text(raw_pred)
    gt_v4 = table_aware_text(raw_gt)

    edit_sim_v1 = edit_similarity(pred_v1, gt_v1)
    char_f1_v2 = char_f1(pred_v2, gt_v2)
    lcs_f1_v1 = lcs_f1(pred_v1, gt_v1)

    loose_edit_sim = max(
        edit_similarity(pred_v2, gt_v2),
        edit_similarity(pred_v3, gt_v3),
    )

    table_aware_sim = edit_similarity(pred_v4, gt_v4)

    text_score = (
        0.25 * edit_sim_v1 + 0.20 * char_f1_v2 + 0.15 * lcs_f1_v1 + 0.20 * loose_edit_sim + 0.20 * table_aware_sim
    )

    bonus = table_compat_bonus(raw_pred, raw_gt, content_sim=max(loose_edit_sim, table_aware_sim))
    rep_pen = repetition_penalty(pred_v4)
    len_pen = length_penalty(pred_v4, gt_v4)

    text_score = text_score + bonus
    text_score = text_score * rep_pen * len_pen
    text_score = max(0.0, min(1.0, text_score))

    return {'text_score': float(text_score)}


# =========================================================
# Part 3. Joint Evaluation
# =========================================================


def evaluate_dataset(
    input_file: str,
    normalize_text_flag: bool = True,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
    blank_reward: Optional[float] = 0.3,
    pred_to_gt_weight: float = 0.5,
    eps: float = 1e-9,
    to_percent: bool = True,
) -> pd.DataFrame:
    """
    Run joint evaluation over an entire JSONL dataset:
    - diou_score
    - text_score
    - overall_score = 0.7 * diou_score + 0.3 * text_score
    """
    sample_results = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            completion = data.get('completion', '')
            solution = data.get('solution', '')

            layout_metrics = evaluate_layout_sample(
                completion=completion,
                solution=solution,
                normalize_text_flag=normalize_text_flag,
                min_reward=min_reward,
                max_reward=max_reward,
                blank_reward=blank_reward,
                pred_to_gt_weight=pred_to_gt_weight,
                eps=eps,
            )

            text_metrics = evaluate_text_sample(
                completion=completion,
                solution=solution,
            )

            diou_score = layout_metrics['diou_score']
            text_score = text_metrics['text_score']
            overall_score = 0.7 * diou_score + 0.3 * text_score

            sample_results.append({
                'diou_score': float(diou_score),
                'text_score': float(text_score),
                'overall_score': float(overall_score),
            })

    if not sample_results:
        result = {
            'diou_score': 0.0,
            'text_score': 0.0,
            'overall_score': 0.0,
        }
        return pd.DataFrame([result])

    df_samples = pd.DataFrame(sample_results)

    result = {
        'diou_score': df_samples['diou_score'].mean(),
        'text_score': df_samples['text_score'].mean(),
        'overall_score': df_samples['overall_score'].mean(),
    }

    if to_percent:
        result = {k: v * 100.0 for k, v in result.items()}

    return pd.DataFrame([result])


def main():
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
