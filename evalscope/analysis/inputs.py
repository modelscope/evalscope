"""Load persisted analysis inputs once, with explicit coverage and cache dependencies."""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from evalscope.api.analysis import FailureCase, SampleKey, SampleRef
from evalscope.api.analysis.config import TraceAnalysisConfig
from evalscope.api.evaluator.cache import ReviewResult
from evalscope.report import Report
from evalscope.report.diagnosis import iter_review_files
from evalscope.utils.io_utils import yaml_to_dict

from .trace.analyzer import analyze_records
from .trace.reader import load_review_rows


class SampleInput(BaseModel):
    """The same sample identity joins reviews, predictions and all rendered views."""

    key: SampleKey
    record: ReviewResult
    case: FailureCase
    stop_reason: Optional[str] = None

    @property
    def recorded_error(self) -> Optional[tuple[str, str]]:
        """Return a recorded scoring or execution problem, without judging the answer."""
        score = self.record.sample_score.score
        metadata = score.metadata or {}
        if self.case.score_status not in ('success', 'fallback', 'degraded'):
            kind = 'judge_failure'
            reason = (
                (score.judge_summary.error if score.judge_summary else None)
                or score.explanation
                or metadata.get('error')
            )
        else:
            execution = metadata.get('execution_result')
            if not isinstance(execution, dict) or execution.get('status') != 'error':
                return None
            kind, reason = 'environment_failure', execution.get('error')
        return (kind, reason) if isinstance(reason, str) and reason.strip() else None


class BenchmarkInput(BaseModel):
    """Only consumed settings are retained; credentials never enter diagnostic outputs."""

    samples: List[SampleInput] = Field(default_factory=list)
    report: Optional[Report] = None
    repeats: int = 1
    unreadable: int = 0
    unreadable_predictions: int = 0

    @property
    def is_multiple_choice(self) -> bool:
        """Recognize the persisted single-letter target format without guessing choice content."""
        gold = [s.record.target for s in self.samples]
        return bool(gold) and sum(bool(g and len(g) == 1 and is_choice_letter(g[0])) for g in gold) / len(gold) >= 0.9


def is_choice_letter(text: str) -> bool:
    """Check only the answer format; the available choice set may not be persisted."""
    value = text.strip()
    return len(value) == 1 and value.isascii() and value.isupper()


def unsupported_content(record: ReviewResult) -> List[str]:
    """List content omitted from text analysis without exposing media or protected reasoning."""
    return sorted(
        {
            'redacted_reasoning' if part.type == 'reasoning' else part.type
            for message in record.messages
            if isinstance(message.content, list)
            for part in message.content
            if part.type not in ('text', 'reasoning') or (part.type == 'reasoning' and part.redacted)
        }
    )


def load_inputs(run_dir: str, model: str, benchmark: str, config: TraceAnalysisConfig) -> BenchmarkInput:
    """Read JSONL by physical lines; Unicode line separators inside strings are valid data."""
    root = Path(run_dir)
    result = BenchmarkInput()
    settings = root / 'configs' / 'task_config.yaml'
    if settings.is_file():
        task = yaml_to_dict(str(settings)) or {}
        result.repeats = task.get('repeats') or 1
    report = root / 'reports' / model / f'{benchmark}.json'
    directions = {}
    correctness_metrics = set()
    if report.is_file():
        result.report = Report.model_validate_json(report.read_text())
        for name in {metric.identity.name for metric in result.report.metrics}:
            metrics = [metric for metric in result.report.metrics if metric.identity.name == name]
            declared = {metric.semantics.direction.value for metric in metrics}
            directions[name] = next(iter(declared)) if len(declared) == 1 else 'none'
            means = [metric for metric in metrics if metric.identity.aggregation == 'mean']
            if means and all(
                metric.semantics.semantic_id in ('quality.accuracy.ratio', 'quality.exact_match.ratio')
                for metric in means
            ):
                correctness_metrics.add(name)
    seen = set()
    for file in iter_review_files(run_dir):
        if file.model != model or file.benchmark != benchmark:
            continue
        records, bad = load_review_rows(file)
        result.unreadable += bad
        refs = [
            SampleRef(run_dir=run_dir, model=model, benchmark=benchmark, subset=file.subset, index=r.index)
            for r in records
        ]
        predictions = {}
        prediction_path = root / 'predictions' / model / Path(file.path).name
        if prediction_path.is_file():
            with prediction_path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        index = row['index']
                        if index in predictions:
                            predictions[index] = None
                            raise ValueError('Ambiguous prediction identity')
                        output = row.get('model_output') or {}
                        choices = output.get('choices') or []
                        predictions[index] = choices[0].get('stop_reason') if choices else None
                    except (ValueError, KeyError, TypeError, AttributeError):
                        result.unreadable_predictions += 1
        cases = analyze_records(zip(refs, records), config, directions, correctness_metrics)
        for ref, record, case in zip(refs, records, cases):
            identity = (ref.subset, ref.index)
            if identity in seen:
                raise ValueError(f'Duplicate review identity: {identity}')
            seen.add(identity)
            result.samples.append(
                SampleInput(
                    key=SampleKey(subset=ref.subset, index=ref.index),
                    record=record,
                    case=case,
                    stop_reason=predictions.get(record.index),
                )
            )
    return result
