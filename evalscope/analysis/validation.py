"""Offline comparison with separately prepared labels; no model-generated explanation certifies itself."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evalscope.api.analysis import Attribution, SampleDiagnosis, SampleKey
from evalscope.api.analysis.taxonomy import ISSUE_NAMES


class DiagnosticLabel(BaseModel):
    """Blind label for a frozen sample, with explicit detector decisions and semantic evidence review."""

    model_config = ConfigDict(extra='forbid')
    key: SampleKey
    failure_type: str
    attribution: Attribution
    root_cause_step: Optional[int] = None
    rule_decisions: Dict[str, bool] = Field(default_factory=dict)
    evidence_supported: Optional[bool] = None
    notes: str = ''
    annotator: str
    user_reviewed: bool = False

    @field_validator('failure_type')
    @classmethod
    def known_type(cls, value: str) -> str:
        """A display label cannot silently become a ground-truth class."""
        if value not in ISSUE_NAMES:
            raise ValueError('Labels must use taxonomy IDs')
        return value


def evaluate_labels(samples: List[SampleDiagnosis], labels: List[DiagnosticLabel]) -> dict:
    """Report proposal accuracy separately from citation availability and semantic review coverage."""
    predictions = {(s.key.subset, s.key.index): s for s in samples}
    expected = {(label.key.subset, label.key.index): label for label in labels}
    if len(expected) != len(labels) or len(predictions) != len(samples):
        raise ValueError('Duplicate sample identities; validate one run/model/benchmark at a time')
    confusion = Counter()
    rules = {}
    correct = proposal_correct = explained = matched = root_correct = supported = semantic = 0
    roots = sum(label.root_cause_step is not None for label in labels)
    for key, label in expected.items():
        sample = predictions.get(key)
        if sample is None:
            continue
        matched += 1
        ok = sample.llm_status == 'ok' and sample.llm is not None
        predicted = sample.llm.failure_type if ok else 'unexplained'
        correct += predicted == label.failure_type
        proposal_correct += ok and predicted == label.failure_type
        explained += ok
        confusion[f'{label.attribution.value}->{sample.llm.attribution.value if ok else "unknown"}'] += 1
        if label.root_cause_step is not None:
            root_correct += ok and sample.llm.root_cause_step == label.root_cause_step
        if label.evidence_supported is not None:
            semantic += 1
            supported += label.evidence_supported
        fired = {f.rule_id for f in sample.findings}
        high_fired = {f.rule_id for f in sample.findings if f.confidence == 'high'}
        for rule, truth in label.rule_decisions.items():
            counts = rules.setdefault(rule, {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'high_tp': 0, 'high_fp': 0})
            counts['tp' if truth and rule in fired else 'fn' if truth else 'fp' if rule in fired else 'tn'] += 1
            if rule in high_fired:
                counts['high_tp' if truth else 'high_fp'] += 1
    for counts in rules.values():
        counts['precision'] = counts['tp'] / (counts['tp'] + counts['fp']) if counts['tp'] + counts['fp'] else None
        high_total = counts['high_tp'] + counts['high_fp']
        counts['high_precision'] = counts['high_tp'] / high_total if high_total else None
        counts['recall'] = counts['tp'] / (counts['tp'] + counts['fn']) if counts['tp'] + counts['fn'] else None
    citations = sum(len(s.llm.evidence) + s.invalid_evidence for s in samples if s.llm)
    valid = sum(len(s.llm.evidence) for s in samples if s.llm)
    return {
        'labeled': len(labels),
        'label_sources': sorted({label.annotator for label in labels}),
        'matched': matched,
        'missing': len(labels) - matched,
        'user_reviewed': sum(label.user_reviewed for label in labels),
        'type_accuracy': correct / len(labels) if labels else None,
        'proposal_coverage': explained / len(labels) if labels else None,
        'proposal_type_accuracy': proposal_correct / explained if explained else None,
        'attribution_confusion': dict(confusion),
        'root_step_accuracy': root_correct / roots if roots else None,
        'root_step_labeled': roots,
        'citation_locatability': valid / citations if citations else None,
        'semantic_reviewed': semantic,
        'evidence_support_rate': supported / semantic if semantic else None,
        'rules': rules,
        'experimental': not explained
        or not labels
        or matched != len(labels)
        or correct / len(labels) < 0.7
        or any(not label.user_reviewed or label.evidence_supported is not True for label in labels)
        or any(c['high_precision'] is not None and c['high_precision'] < 0.9 for c in rules.values()),
    }


def main() -> None:
    """Compare one persisted samples file with blind JSONL labels and save machine-readable metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    def rows(path: str, schema: type) -> list:
        with Path(path).open() as stream:
            return [schema.model_validate_json(line) for line in stream if line.strip()]

    result = evaluate_labels(rows(args.samples, SampleDiagnosis), rows(args.labels, DiagnosticLabel))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
