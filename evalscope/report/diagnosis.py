"""Read diagnostic artifacts and their source identity without importing the analysis engine."""

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from pydantic import BaseModel

from evalscope.api.analysis import DiagnosisReport
from evalscope.api.analysis.schema import AnalysisFactsConfig
from evalscope.utils.io_utils import OutputsStructure, yaml_to_dict
from evalscope.utils.logger import get_logger

logger = get_logger()


class ReviewFile(BaseModel):
    """One ``reviews/<model>/<benchmark>_<subset>.jsonl`` file of a run."""

    run_dir: str
    model: str
    benchmark: str
    subset: str
    path: str


def split_review_name(stem: str, known_benchmarks: List[str]) -> Tuple[str, str]:
    """Split ``<benchmark>_<subset>`` with the longest known benchmark name (names may contain underscores)."""
    for name in sorted(set(known_benchmarks), key=len, reverse=True):
        if stem == name:
            return name, ''
        if stem.startswith(f'{name}_'):
            return name, stem[len(name) + 1 :]
    return stem, ''


def _known_benchmarks(run_dir: str, model: str) -> List[str]:
    names: List[str] = []
    config_path = os.path.join(run_dir, OutputsStructure.CONFIGS_DIR, 'task_config.yaml')
    if os.path.isfile(config_path):
        try:
            names += [name for name in (yaml_to_dict(config_path) or {}).get('datasets') or [] if isinstance(name, str)]
        except Exception as exc:  # noqa: BLE001 - malformed snapshots can fall back to report identities
            logger.warning(f'Cannot read benchmark identities from {config_path}: {type(exc).__name__}')
    report_dir = os.path.join(run_dir, OutputsStructure.REPORTS_DIR, model)
    if os.path.isdir(report_dir):
        names += [name[:-5] for name in os.listdir(report_dir) if name.endswith('.json')]
    return names


def iter_review_files(run_dir: str) -> Iterator[ReviewFile]:
    """Yield the review files of one run dir in a stable order."""
    reviews_root = os.path.join(run_dir, OutputsStructure.REVIEWS_DIR)
    if not os.path.isdir(reviews_root):
        return
    for model in sorted(os.listdir(reviews_root)):
        model_dir = os.path.join(reviews_root, model)
        if not os.path.isdir(model_dir):
            continue
        known = _known_benchmarks(run_dir, model)
        for file_name in sorted(name for name in os.listdir(model_dir) if name.endswith('.jsonl')):
            benchmark, subset = split_review_name(file_name[: -len('.jsonl')], known)
            yield ReviewFile(
                run_dir=run_dir,
                model=model,
                benchmark=benchmark,
                subset=subset,
                path=os.path.join(model_dir, file_name),
            )


def input_files(run_dir: str, model: str, benchmark: str) -> List[Path]:
    """Enumerate consumed source files, including predictions and report metadata."""
    root = Path(run_dir)
    reviews = [Path(f.path) for f in iter_review_files(run_dir) if f.model == model and f.benchmark == benchmark]
    predictions = [root / 'predictions' / model / p.name for p in reviews]
    return sorted(
        {
            p
            for p in [
                *reviews,
                *predictions,
                root / 'reports' / model / f'{benchmark}.json',
                root / 'configs' / 'task_config.yaml',
            ]
            if p.is_file()
        }
    )


def data_fingerprint(run_dir: str, model: str, benchmark: str) -> str:
    """Ignore our own Markdown writeback and unrelated evaluation progress changes."""
    digest = hashlib.sha256()
    for path in input_files(run_dir, model, benchmark):
        digest.update(str(path.relative_to(run_dir)).encode())
        if path.suffix == '.yaml':
            data = yaml_to_dict(str(path)) or {}
            payload = {key: data.get(key) for key in ('datasets', 'repeats')}
            digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
        elif path.suffix == '.json':
            payload = json.loads(path.read_text())
            payload.pop('analysis', None)
            digest.update(json.dumps(payload, sort_keys=True).encode())
        else:
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def diagnosis_dir(run_dir: str, model: str, benchmark: str, level: str = 'facts') -> Path:
    """Return the location for a base or model analysis snapshot."""
    base = Path(run_dir) / 'analysis' / model / benchmark
    return base / 'diagnose' if level == 'diagnose' else base


def load_diagnosis(run_dir: str, model: str, benchmark: str) -> Tuple[str, Optional[DiagnosisReport]]:
    """Read only; never generate diagnostics while serving a report."""
    status, report, _ = load_diagnosis_snapshot(run_dir, model, benchmark)
    return status, report


class DiagnosisSnapshot(DiagnosisReport):
    """On-disk report with the non-model settings needed to reproduce its observations."""

    facts_config: Optional[AnalysisFactsConfig] = None


def load_diagnosis_snapshot(run_dir: str, model: str, benchmark: str) -> Tuple[str, Optional[DiagnosisSnapshot], Path]:
    """Prefer model analysis only while its underlying facts still match."""
    base = diagnosis_dir(run_dir, model, benchmark)
    model_target = diagnosis_dir(run_dir, model, benchmark, 'diagnose')
    model_status, model_report = read_diagnosis_snapshot(run_dir, model, benchmark, model_target)
    fingerprint = model_report.data_fingerprint if model_status == 'ready' else None
    base_status, base_report = read_diagnosis_snapshot(run_dir, model, benchmark, base, fingerprint)
    if model_status == 'ready':
        if (
            base_status == 'ready'
            and base_report.level == 'facts'
            and base_report.facts_config is not None
            and base_report.facts_config != model_report.facts_config
        ):
            return base_status, base_report, base
        return model_status, model_report, model_target
    if base_status == 'ready' or model_status == 'missing':
        return base_status, base_report, base
    return model_status, model_report, model_target


def read_diagnosis_snapshot(
    run_dir: str, model: str, benchmark: str, target: Path, fingerprint: Optional[str] = None
) -> Tuple[str, Optional[DiagnosisSnapshot]]:
    """Read and validate a snapshot in a specific directory."""
    path = target / 'diagnosis.json'
    if not path.is_file():
        return 'missing', None
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return 'error', None
        if payload.get('schema_version', 1) != DiagnosisReport.model_fields['schema_version'].default:
            return 'stale', None
        report = DiagnosisSnapshot.model_validate(payload)
    except (ValueError, OSError):
        return 'error', None
    if report.coverage.analysis_errors:
        return 'error', report
    try:
        fresh = report.data_fingerprint == (fingerprint or data_fingerprint(run_dir, model, benchmark))
        fresh = fresh and (target / 'samples.jsonl').is_file()
        if fresh:
            fresh = hashlib.sha256((target / 'samples.jsonl').read_bytes()).hexdigest() == report.samples_sha256
    except (ValueError, OSError):
        fresh = False
    return ('ready' if fresh else 'stale'), report
