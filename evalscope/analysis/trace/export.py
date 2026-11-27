"""Write analysis results to disk with secret redaction."""

import os
import re
from typing import Any, List

from evalscope.api.analysis import Finding
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


def redact_data(value: Any, secrets: List[str]) -> Any:
    """Redact keys and values before JSON escaping can hide a secret from exact matching."""
    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, dict):
        return {
            redact(key, secrets) if isinstance(key, str) else key: redact_data(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_data(item, secrets) for item in value]
    return value


def export_findings(findings: List[Finding], config: TraceAnalysisConfig) -> List[Finding]:
    """Drop internal error text and cap evidence; the final artifact writer handles redaction."""
    exported = []
    remaining = config.max_evidence_chars_per_case
    for finding in findings:
        evidence = []
        for item in finding.evidence:
            if len(item.excerpt) > remaining:
                break
            evidence.append(item)
            remaining -= len(item.excerpt)
        exported.append(finding.model_copy(update={'text': '', 'evidence': evidence}))
    return exported
