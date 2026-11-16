"""Write analysis results to disk with secret redaction."""

import json
import os
import re
from typing import Dict, List

from evalscope.api.analysis import FailureCase, Finding
from evalscope.api.analysis.config import TraceAnalysisConfig

_SECRET_PATTERNS = [
    re.compile(r'\bsk-[A-Za-z0-9_\-]{16,}'),
    re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}'),
    re.compile(
        r'(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|authorization)(["\'\s]*[:=]["\'\s]*)([^\s"\',;]{8,})'
    ),
]
_SECRET_ENV_NAME = re.compile(r'(?i)(KEY|TOKEN|SECRET|PASSWORD|SANDBOX)')


def _env_secrets() -> List[str]:
    return sorted(
        {value for name, value in os.environ.items() if _SECRET_ENV_NAME.search(name) and value and len(value) >= 16},
        key=len,
        reverse=True,
    )


def redact(text: str, env_secrets: List[str]) -> str:
    """Mask API keys, bearer tokens, ``key=value`` secrets and secret-looking environment values."""
    if not text:
        return text
    for secret in env_secrets:
        if secret in text:
            text = text.replace(secret, '***')
    text = _SECRET_PATTERNS[0].sub('sk-***', text)
    text = _SECRET_PATTERNS[1].sub('Bearer ***', text)
    text = _SECRET_PATTERNS[2].sub(lambda m: f'{m.group(1)}{m.group(2)}***', text)
    return text


def _redact_finding(finding: Finding, secrets: List[str]) -> Dict:
    data = finding.model_dump(mode='json', exclude={'text'})
    for evidence in data['evidence']:
        evidence['excerpt'] = redact(evidence['excerpt'], secrets)
    data['data'] = json.loads(redact(json.dumps(data['data'], ensure_ascii=False), secrets))
    return data


def _cap_evidence(case_dict: Dict, limit: int) -> None:
    used = len(case_dict.get('task_excerpt') or '')
    for finding in case_dict['findings']:
        kept = []
        for evidence in finding['evidence']:
            size = len(evidence['excerpt'])
            if used + size > limit:
                break
            used += size
            kept.append(evidence)
        finding['evidence'] = kept


def case_to_dict(case: FailureCase, config: TraceAnalysisConfig, secrets: List[str]) -> Dict:
    """Serialize one failure case with redaction and the per-case evidence budget."""
    data = case.model_dump(mode='json', exclude={'findings'})
    data['task_excerpt'] = redact(data['task_excerpt'], secrets)
    data['findings'] = [_redact_finding(finding, secrets) for finding in case.findings]
    _cap_evidence(data, config.max_evidence_chars_per_case)
    return data
