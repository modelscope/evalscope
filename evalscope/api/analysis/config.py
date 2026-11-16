"""Configuration of the trace analyzer, with reviewable default rules."""

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_ERROR_PATTERNS: Dict[str, str] = {
    'traceback': r'Traceback \(most recent call last\)',
    'exception_prefix': r'(?m)^\s*[A-Z][A-Za-z]*(?:Error|Exception):\s',
    'validation_error': r'\b\d+ validation errors? for\b|\bValidationError\b',
    'exit_code': r'(?i)\b(?:exit code|returncode|exit status)[:= ]+[1-9]\d*',
    'not_found_tool': r'(?i)(?:is not registered|unknown tool|invalid tool \S+ provided|is not a valid tool)',
    'empty_function_name': r'(?i)empty function name provided',
}
"""Patterns that mark a tool observation as failed when no structured error is present."""

DEFAULT_GUARD_PATTERNS: Dict[str, str] = {
    'blocked_by_policy': r'(?i)\bblocked by (?:the )?(?:policy|guard|guardrail|safety filter)\b',
    'denied_by': r'(?i)\b(?:denied|rejected) by (?:the )?(?:policy|guard|guardrail|administrator|approver)\b',
    'requires_approval': r'(?i)\brequires? (?:human |user )?approval\b',
    'not_permitted': r'(?i)\boperation not permitted by policy\b',
}

DEFAULT_ENVIRONMENT_PATTERNS: Dict[str, str] = {
    'timeout': r'(?i)\bTimeoutError\b|\bTimeoutException\b|\btimed out\b|\bdeadline exceeded\b',
    'sandbox': r'(?i)\bsandbox (?:error|not found|was killed|timeout)\b|\benvd\b|\bOOMKilled\b|No space left on device',
    'network': (
        r'(?i)\bConnection(?:Error| refused| reset| aborted)\b|\bReadTimeout\b|Name or service not known'
        r'|\bService Unavailable\b|\bBad Gateway\b|\bGateway Timeout\b|\brate limit(?:ed)?\b|\bHTTP(?:Error)? 5\d\d\b'
    ),
}

DEFAULT_SIDE_EFFECT_VERBS: List[str] = [
    'send',
    'delete',
    'remove',
    'transfer',
    'pay',
    'update',
    'create',
    'write',
    'post',
    'share',
    'invite',
    'book',
    'reserve',
    'cancel',
    'schedule',
    'reschedule',
    'append',
    'modify',
    'move',
    'rename',
    'upload',
    'add',
    'edit',
    'purchase',
    'buy',
    'order',
    'publish',
    'forward',
    'reply',
    'grant',
    'revoke',
    'kick',
    'ban',
]

DEFAULT_SHELL_TOOLS: List[str] = ['bash', 'shell', 'execute_bash', 'run_command', 'terminal', 'execute', 'sh']

DEFAULT_SHELL_RISK_PATTERNS: Dict[str, str] = {
    'rm_recursive_force': r'\brm\s+(?:-[a-zA-Z]*[rR][a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*[rR]|--recursive\s+--force|-r\s+-f|-f\s+-r)',
    'http_upload': r'\b(?:curl|wget)\b[^\n]*(?:\s-X\s*POST|\s--data|\s-d\s|\s-F\s|--upload-file|\s-T\s|--post-data)',
    'pipe_to_shell': r'\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b',
    'chmod_world_writable': r'\bchmod\s+(?:-R\s+)?0?777\b',
    'git_push': r'\bgit\s+push\b',
    'disk_destroy': r'\b(?:mkfs(?:\.\w+)?|dd\s+if=[^\n]*\bof=/dev/)',
    'system_path_write': r'(?:>|\btee\b)\s*/(?:etc|usr|bin|sbin|boot)/',
    'firewall_off': r'\b(?:ufw\s+disable|iptables\s+-F|setenforce\s+0)\b',
    'kill_all': r'\b(?:kill\s+-9\s+-1|pkill\s+-9\s+-f\s+\.)',
}

DEFAULT_REFUSAL_PATTERNS: List[str] = [
    r"(?i)\bI(?:'m| am) (?:sorry|unable|not able)\b.{0,80}\b(?:can(?:no|')t|cannot|unable|won't|will not)\b",
    r"(?i)\bI (?:can(?:no|')t|cannot|won't|will not) (?:help|assist|comply|do that|perform|complete)\b",
    r'(?i)\bI must (?:decline|refuse)\b',
    r'抱歉.{0,20}(?:无法|不能)',
    r'我(?:无法|不能)(?:帮助|协助|完成|执行)',
]


class TraceAnalysisConfig(BaseModel):
    """Thresholds and patterns used by the trace analyzer.

    Every rule is a plain regex or word list so that a reviewer can audit the
    behavior; there is no learned component.
    """

    model_config = ConfigDict(extra='forbid')

    repeat_min_count: int = Field(default=3, ge=2)
    no_progress_tail: int = Field(default=4, ge=2)
    oscillation_min_length: int = Field(default=4, ge=4)
    error_patterns: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_ERROR_PATTERNS))
    guard_patterns: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_GUARD_PATTERNS))
    environment_patterns: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_ENVIRONMENT_PATTERNS))
    side_effect_verbs: List[str] = Field(default_factory=lambda: list(DEFAULT_SIDE_EFFECT_VERBS))
    shell_tool_names: List[str] = Field(default_factory=lambda: list(DEFAULT_SHELL_TOOLS))
    shell_risk_patterns: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SHELL_RISK_PATTERNS))
    refusal_patterns: List[str] = Field(default_factory=lambda: list(DEFAULT_REFUSAL_PATTERNS))
    taint_min_string_len: int = Field(default=8, ge=4)
    error_scan_chars: int = Field(default=2000, ge=100)
    outcome_thresholds: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    outcome_directions: Dict[str, Literal['higher_is_better', 'lower_is_better', 'none']] = Field(default_factory=dict)
    """Threshold comparison follows each benchmark's declared metric direction."""

    max_evidence_chars_per_case: int = Field(default=4000, ge=200)
    evidence_excerpt_chars: int = Field(default=500, ge=80)
    use_providers: bool = True


__all__ = ['TraceAnalysisConfig']
