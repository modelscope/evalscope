# Copyright (c) Alibaba, Inc. and its affiliates.

import base64
import json
import math
from typing import Any, Dict

RESULT_SENTINEL = '__EVALSCOPE_OMNIDOCBENCH_V1_6_RESULT__='

PAGE_METRICS = (
    'text_block_Edit_dist',
    'display_formula_Edit_dist',
    'display_formula_CDM',
    'table_TEDS',
    'table_TEDS_structure_only',
    'table_Edit_dist',
    'reading_order_Edit_dist',
)


def build_scoring_program(annotation: Dict[str, Any], image_name: str, prediction: str) -> str:
    """Build the single-page program executed inside the pinned official image."""
    payload = base64.b64encode(
        json.dumps(
            {
                'annotation': annotation,
                'image_name': image_name,
                'prediction': prediction,
            },
            ensure_ascii=False,
        ).encode('utf-8')
    ).decode('ascii')

    return f'''import base64
import json
import math
import os
import signal
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/workspace")
from src.core.pipeline import run_config

def handle_termination(*_):
    raise TimeoutError("sandbox timeout")

signal.signal(signal.SIGTERM, handle_termination)
payload = json.loads(base64.b64decode("{payload}").decode("utf-8"))
work_dir = Path(tempfile.mkdtemp(prefix="evalscope-omnidocbench-v1.6-"))
previous_dir = Path.cwd()
try:
    os.environ["CDM_SAVE_VIS"] = "0"
    gt_path = work_dir / "ground_truth.json"
    prediction_dir = work_dir / "predictions"
    prediction_dir.mkdir()
    (work_dir / "result").mkdir()
    gt_path.write_text(json.dumps([payload["annotation"]], ensure_ascii=False), encoding="utf-8")
    prediction_path = prediction_dir / (Path(payload["image_name"]).stem + ".md")
    prediction_path.write_text(payload["prediction"], encoding="utf-8")

    config = {{
        "end2end_eval": {{
            "metrics": {{
                "text_block": {{"metric": ["Edit_dist"]}},
                "display_formula": {{"metric": ["Edit_dist", "CDM"], "cdm_workers": 1}},
                "table": {{"metric": ["TEDS", "Edit_dist"], "teds_workers": 1}},
                "reading_order": {{"metric": ["Edit_dist"]}},
            }},
            "dataset": {{
                "dataset_name": "end2end_dataset",
                "ground_truth": {{"data_path": str(gt_path)}},
                "prediction": {{"data_path": str(prediction_dir)}},
                "match_method": "quick_match",
                "match_workers": 1,
                "quick_match_truncated_timeout_sec": 300,
                "match_timeout_sec": 420,
                "timeout_fallback_max_chunk_span": 10,
                "timeout_fallback_order_penalty": 0.10,
            }},
        }}
    }}

    os.chdir(work_dir)
    run_config(config)
    result_path = work_dir / "result" / "predictions_quick_match_metric_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    def read_metric(*path):
        value = result
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    metrics = {{
        "text_block_Edit_dist": read_metric("text_block", "all", "Edit_dist", "ALL_page_avg"),
        "display_formula_Edit_dist": read_metric("display_formula", "all", "Edit_dist", "ALL_page_avg"),
        "display_formula_CDM": read_metric("display_formula", "page", "CDM", "ALL"),
        "table_TEDS": read_metric("table", "page", "TEDS", "ALL"),
        "table_TEDS_structure_only": read_metric("table", "page", "TEDS_structure_only", "ALL"),
        "table_Edit_dist": read_metric("table", "all", "Edit_dist", "ALL_page_avg"),
        "reading_order_Edit_dist": read_metric("reading_order", "all", "Edit_dist", "ALL_page_avg"),
    }}
    metrics = {{name: value for name, value in metrics.items() if value is not None}}
    print("{RESULT_SENTINEL}" + json.dumps(metrics, allow_nan=False, sort_keys=True))
finally:
    os.chdir(previous_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
'''


def parse_scoring_result(result: Dict[str, Any]) -> Dict[str, float]:
    """Parse and validate the metric sentinel returned by the sandbox program."""
    if result.get('status') != 'success':
        detail = str(result.get('error') or result.get('output') or 'unknown sandbox error')
        raise RuntimeError(f'Official OmniDocBench v1.6 scoring failed: {detail[-2000:]}')

    output = str(result.get('output') or '')
    payload = None
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            payload = line[len(RESULT_SENTINEL) :]
            break
    if payload is None:
        raise RuntimeError('Official OmniDocBench v1.6 scoring did not return a metric result.')

    try:
        values = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError('Official OmniDocBench v1.6 scoring returned invalid JSON.') from error
    if not isinstance(values, dict) or not values:
        raise RuntimeError('Official OmniDocBench v1.6 scoring returned no page metrics.')

    unexpected = sorted(set(values) - set(PAGE_METRICS))
    if unexpected:
        raise RuntimeError(f'Official OmniDocBench v1.6 scoring returned unexpected metrics: {unexpected}.')

    metrics = {}
    for name, value in values.items():
        if isinstance(value, bool):
            raise RuntimeError(f'Official OmniDocBench v1.6 metric {name} is not numeric.')
        try:
            resolved = float(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f'Official OmniDocBench v1.6 metric {name} is not numeric.') from error
        if not math.isfinite(resolved):
            raise RuntimeError(f'Official OmniDocBench v1.6 metric {name} is not finite.')
        if not 0.0 <= resolved <= 1.0:
            raise RuntimeError(f'Official OmniDocBench v1.6 metric {name} is outside the expected 0-1 range.')
        metrics[name] = resolved
    return metrics
