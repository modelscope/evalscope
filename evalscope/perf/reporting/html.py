from __future__ import annotations

import html
import json
from typing import Optional

from evalscope.perf.domain.result import PerfSuiteResult


def write_html_report(result: PerfSuiteResult, path: Optional[str]) -> None:
    """Write a self-contained report directly from a typed suite result."""
    if path is None:
        return
    rows = []
    for run in result.runs:
        rows.append(
            '<tr>'
            f'<td>{html.escape(run.run_spec.load_id)}</td>'
            f'<td>{run.summary.total}</td>'
            f'<td>{run.summary.succeeded}</td>'
            f'<td>{run.summary.failed}</td>'
            f'<td>{run.summary.dropped}</td>'
            f'<td>{run.summary.request_throughput:.3f}</td>'
            f'<td>{run.summary.averages.get("latency", 0):.4f}</td>'
            f'<td>{run.summary.success_rate:.1f}%</td>'
            '</tr>'
        )
    raw = html.escape(json.dumps(result.model_dump(mode='json'), ensure_ascii=False, indent=2))
    document = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>EvalScope Perf {html.escape(result.run_id)}</title>
<style>body{{font-family:system-ui;margin:2rem;color:#1f2937}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d1d5db;padding:.5rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}
pre{{background:#f3f4f6;padding:1rem;overflow:auto}}</style></head>
<body><h1>EvalScope Performance Report</h1><p>Run: {html.escape(result.run_id)}</p>
<table><thead><tr><th>Load</th><th>Total</th><th>Succeeded</th><th>Failed</th><th>Dropped</th>
<th>RPS</th><th>Avg latency (s)</th><th>Success</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Typed result</h2><pre>{raw}</pre></body></html>'''
    with open(path, 'w', encoding='utf-8') as file:
        file.write(document)
