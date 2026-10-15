"""Render the risk-coverage points preserved in an EvalScope decision report."""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Write a standalone interactive HTML chart without contacting an external service."""
    import plotly.graph_objects as go

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    figure = go.Figure()
    for metric in report['metrics']:
        identity = metric['identity']
        if identity['name'] != 'aurc':
            continue
        dimensions = identity['dimensions']
        basis = dimensions['confidence']
        for category in metric['categories']:
            for subset in category['subsets']:
                points = (subset.get('metadata') or {}).get(f'risk_coverage_{basis}', [])
                if points:
                    label = f'{subset["name"]} / {dimensions["kind"]} / {basis}'
                    if 'levels' in dimensions:
                        label += f' / {dimensions["levels"]} levels'
                    figure.add_scatter(x=[point['coverage'] for point in points],
                                       y=[point['risk'] for point in points], mode='lines', name=label)
    if not figure.data:
        raise ValueError('This report has no risk-coverage observations.')
    figure.update_layout(title=f'Risk–coverage: {report["model_name"]}',
                         xaxis_title='Coverage', yaxis_title='Risk (classification error / normalized MAE)',
                         xaxis_range=[0, 1], yaxis_range=[0, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs=True)


if __name__ == '__main__':
    main()
