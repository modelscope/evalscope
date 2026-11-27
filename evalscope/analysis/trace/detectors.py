"""Rule-based trace detectors.

Every detector reads one :class:`DetectionContext` and returns findings. They run in a fixed
order: tool failures first (repetition reads the failed positions), then repetition, abnormal
endings, side effects with argument provenance, and refusals (which read the earlier findings).
All rules are plain regexes or word lists from :class:`TraceAnalysisConfig`.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Pattern, Set, Tuple

from evalscope.api.agent.constants import LoopMessages, SubmissionSources
from evalscope.api.analysis import (
    Confidence,
    Evidence,
    Finding,
    FindingCategory,
    ProviderSignals,
    TraceToolCall,
    TraceView,
)
from evalscope.api.analysis.config import TraceAnalysisConfig

TOOL_FAILURE = 'tool_failure@1'
REPETITION = 'repetition@1'
TERMINATION = 'termination@1'
SIDE_EFFECT = 'side_effect@2'
REFUSAL = 'refusal@1'

_WS = re.compile(r'\s+')
_EDGE_PUNCT = ' \t\n.,;:!?"\'`()[]{}<>'
_SHELL_ARG_NAMES = frozenset({'command', 'cmd', 'script', 'timeout', 'cwd'})
_NONZERO_EXIT = re.compile(
    r'<returncode>\s*(-?[1-9]\d*)\s*</returncode>|(?i:\b(?:exit code|returncode|exit status)[:= ]+(-?[1-9]\d*))'
)
_NAME_SPLIT = re.compile(r'[_\-\s.]+|(?<=[a-z0-9])(?=[A-Z])')
# Dates, years and clock times recur across unrelated texts and carry no provenance signal.
_LOW_INFO_ATOM = re.compile(
    r'^(?:(?:19|20)\d{2}(?:[-/.]\d{1,2}){0,2}(?:[ t]\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?'
    r'|\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)\d{2})$'
)
ATOM_PATTERNS: Dict[str, Pattern[str]] = {
    'email': re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+'),
    'url': re.compile(r'https?://[^\s"\'<>]+|\bwww\.[^\s"\'<>]+'),
    'iban_like': re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{8,30}\b'),
    'number': re.compile(r'(?<![\w.])\d{4,}(?![\w])'),
    'path': re.compile(r'(?:/[\w.-]+){2,}'),
}
_LOOP_ERROR_SUBTYPES = {
    LoopMessages.MAX_STEPS_EXCEEDED: 'max_steps_exceeded',
    LoopMessages.MODEL_CONTEXT_OVERFLOW: 'context_overflow',
}
_STOP_REASON_SUBTYPES = {
    'model_length': 'context_overflow',
    'content_filter': 'content_filter',
    'max_steps': 'max_steps_exceeded',
}
_USABLE_STATUSES = frozenset({'success', 'fallback', 'degraded'})


# ---------------------------------------------------------------------------
# Context and helpers
# ---------------------------------------------------------------------------


def compile_patterns(patterns: Dict[str, str]) -> Dict[str, Pattern[str]]:
    """Compile a ``{pattern_id: regex}`` mapping, failing loudly on a bad user pattern."""
    compiled: Dict[str, Pattern[str]] = {}
    for pattern_id, pattern in patterns.items():
        try:
            compiled[pattern_id] = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f'Invalid trace analysis pattern {pattern_id!r}: {exc}') from exc
    return compiled


@dataclass
class CompiledRules:
    """Regexes compiled once per analysis."""

    error: Dict[str, Pattern[str]]
    guard: Dict[str, Pattern[str]]
    environment: Dict[str, Pattern[str]]
    shell_risk: Dict[str, Pattern[str]]
    refusal: List[Pattern[str]]

    @classmethod
    def from_config(cls, config: TraceAnalysisConfig) -> 'CompiledRules':
        return cls(
            error=compile_patterns(config.error_patterns),
            guard=compile_patterns(config.guard_patterns),
            environment=compile_patterns(config.environment_patterns),
            shell_risk=compile_patterns(config.shell_risk_patterns),
            refusal=list(compile_patterns({str(i): p for i, p in enumerate(config.refusal_patterns)}).values()),
        )


@dataclass
class DetectionContext:
    """Everything a detector may read for one sample."""

    view: TraceView
    signals: ProviderSignals
    config: TraceAnalysisConfig
    rules: CompiledRules
    failed_positions: Set[int] = field(default_factory=set)
    """Positions of failed tool calls, filled by :func:`detect_tool_failures`."""

    def finding(
        self,
        category: FindingCategory,
        subtype: str,
        detector: str,
        confidence: Confidence,
        positions: Optional[List[int]] = None,
        evidence: Optional[List[Evidence]] = None,
        data: Optional[Dict[str, Any]] = None,
        text: str = '',
        first_position: Optional[int] = None,
    ) -> Finding:
        """Build a finding for this sample; ``first_position`` defaults to the smallest position."""
        positions = sorted(set(positions or []))
        return Finding(
            category=category,
            subtype=subtype,
            positions=positions,
            first_position=first_position if first_position is not None else (positions[0] if positions else None),
            confidence=confidence,
            detector=detector,
            evidence=evidence or [],
            data=data or {},
            text=text,
        )


def norm(text: str) -> str:
    """Lower-case, collapse whitespace and strip edge punctuation for containment checks."""
    return _WS.sub(' ', text.lower()).strip(_EDGE_PUNCT)


def excerpt(text: Optional[str], limit: int, around: Optional[int] = None) -> str:
    """Return at most ``limit`` characters of ``text``, centred on ``around`` when given."""
    if not text:
        return ''
    if len(text) <= limit:
        return text
    if around is None:
        return text[:limit] + '…'
    start = max(0, around - limit // 2)
    end = min(len(text), start + limit)
    return ('…' if start > 0 else '') + text[start:end] + ('…' if end < len(text) else '')


def iter_strings(value: Any) -> Iterator[str]:
    """Yield every string nested in a JSON-like value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_strings(item)


def shell_command(call: TraceToolCall) -> Optional[str]:
    """Return the command string of a shell-like call, if any."""
    for key in ('command', 'cmd', 'script'):
        value = call.arguments.get(key)
        if isinstance(value, str):
            return value
    return None


def is_shell_call(call: TraceToolCall, shell_tools: List[str]) -> bool:
    """A call is a shell call when its tool is a known shell tool or it only carries command-style arguments."""
    if call.name.lower() in {name.lower() for name in shell_tools}:
        return True
    return shell_command(call) is not None and set(call.arguments) <= _SHELL_ARG_NAMES


def call_signature(call: TraceToolCall, shell_tools: List[str]) -> str:
    """Canonical ``name(sorted JSON args)`` signature; shell commands have their whitespace collapsed."""
    arguments: Dict[str, Any] = dict(call.arguments)
    if is_shell_call(call, shell_tools):
        for key in ('command', 'cmd'):
            if isinstance(arguments.get(key), str):
                arguments[key] = _WS.sub(' ', arguments[key]).strip()
    try:
        return f'{call.name}({json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)})'
    except (TypeError, ValueError):
        return f'{call.name}({sorted(arguments.items())!r})'


def call_text(call: TraceToolCall) -> str:
    try:
        return f'{call.name}({json.dumps(call.arguments, ensure_ascii=False, default=str)})'
    except (TypeError, ValueError):
        return f'{call.name}({call.arguments!r})'


def name_tokens(name: str) -> Set[str]:
    """Split a tool name such as ``sendMoney`` or ``send_money`` into lower-case tokens."""
    return {token.lower() for token in _NAME_SPLIT.split(name) if token}


def _informative(atom: str) -> bool:
    return len(atom) >= 4 and _LOW_INFO_ATOM.match(atom) is None


def extract_atoms(arguments: Any, min_string_len: int) -> List[Tuple[str, str]]:
    """Return ``(kind, normalized value)`` atoms of the string arguments: identifiers and long whole strings."""
    atoms: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for value in iter_strings(arguments):
        for kind, pattern in ATOM_PATTERNS.items():
            for match in pattern.finditer(value):
                atom = norm(match.group(0))
                if _informative(atom) and atom not in seen:
                    seen.add(atom)
                    atoms.append((kind, atom))
        whole = norm(value)
        if len(whole) >= min_string_len and _informative(whole) and whole not in seen:
            seen.add(whole)
            atoms.append(('string', whole))
    return atoms


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def detect_tool_failures(ctx: DetectionContext) -> List[Finding]:
    """One finding per failed call: provider flag > parse error > structured error > shell exit code > text."""
    findings: List[Finding] = []
    limit = ctx.config.evidence_excerpt_chars

    def fail(
        call: TraceToolCall,
        subtype: str,
        confidence: Confidence,
        text: str,
        source: str,
        around: Optional[int] = None,
        **data: Any,
    ) -> None:
        prefix = f'{call.name}: '
        evidence = Evidence(
            message_index=call.result_message_index,
            position=call.position,
            excerpt=prefix + excerpt(text, max(40, limit - len(prefix)), around=around),
        )
        findings.append(
            ctx.finding(
                FindingCategory.TOOL_FAILURE,
                subtype,
                TOOL_FAILURE,
                confidence,
                [call.position],
                [evidence],
                {'tool': call.name, 'source': source, **data},
                text,
            )
        )
        ctx.failed_positions.add(call.position)

    for call in ctx.view.tool_calls:
        override = ctx.signals.tool_status.get(call.position)
        if override is not None:
            if override.failed:
                fail(
                    call,
                    f'provider:{override.error_type or "failed"}',
                    Confidence.HIGH,
                    override.message or call.result_text or '',
                    'provider',
                )
            continue
        if call.parse_error:
            fail(call, 'argument_parse_error', Confidence.HIGH, call.parse_error, 'structured')
            continue
        if call.error_type and call.error_source == 'structured':
            fail(
                call,
                f'structured:{call.error_type}',
                Confidence.HIGH,
                call.error_message or call.result_text or '',
                'structured',
            )
            continue
        scanned = (call.result_text or '')[: ctx.config.error_scan_chars]
        if is_shell_call(call, ctx.config.shell_tool_names):
            # Program output (test failures, source code, logs) routinely contains tracebacks and error words,
            # so for shell tools only a non-zero exit status counts, at low confidence.
            match = _NONZERO_EXIT.search(scanned)
            if match is not None:
                fail(
                    call,
                    'command_nonzero_exit',
                    Confidence.LOW,
                    scanned,
                    'text_pattern',
                    match.start(),
                    exit_code=int(match.group(1) or match.group(2)),
                )
            continue
        for pattern_id, pattern in ctx.rules.error.items():
            match = pattern.search(scanned)
            if match is not None:
                fail(call, f'text_pattern:{pattern_id}', Confidence.MEDIUM, scanned, 'text_pattern', match.start())
                break

    for event in ctx.view.events:
        if event.type == 'error' and event.source == 'parse':
            findings.append(
                ctx.finding(
                    FindingCategory.TOOL_FAILURE,
                    'model_output_parse_error',
                    TOOL_FAILURE,
                    Confidence.HIGH,
                    evidence=[Evidence(excerpt=excerpt(event.message, limit))],
                    data={'step': event.step},
                    text=event.message or '',
                )
            )
    return findings


def detect_repetition(ctx: DetectionContext) -> List[Finding]:
    """Retries after errors, A-B oscillation, identical repeats and non-progressing tails."""
    calls = ctx.view.tool_calls
    if len(calls) < 2:
        return []
    signatures = [call_signature(call, ctx.config.shell_tool_names) for call in calls]
    findings: List[Finding] = []
    covered: Set[int] = set()

    def add(subtype: str, positions: List[int]) -> None:
        tools = sorted({calls[p].name for p in positions})
        evidence = [
            Evidence(
                position=positions[0], excerpt=excerpt(signatures[positions[0]], ctx.config.evidence_excerpt_chars)
            )
        ]
        findings.append(
            ctx.finding(
                FindingCategory.REPETITION,
                subtype,
                REPETITION,
                Confidence.MEDIUM,
                positions,
                evidence,
                {'tool': tools[0] if len(tools) == 1 else tools, 'count': len(positions)},
            )
        )

    position = 1
    while position < len(calls):
        if signatures[position] == signatures[position - 1] and (position - 1) in ctx.failed_positions:
            run = [position - 1, position]
            while (
                run[-1] + 1 < len(calls)
                and signatures[run[-1] + 1] == signatures[position]
                and run[-1] in ctx.failed_positions
            ):
                run.append(run[-1] + 1)
            add('retry_unchanged_after_error', run)
            covered.update(run)
            position = run[-1] + 1
        else:
            position += 1

    start = 0
    while start + 1 < len(calls):
        if signatures[start] == signatures[start + 1]:
            start += 1
            continue
        end = start + 2
        while end < len(calls) and signatures[end] == signatures[end - 2]:
            end += 1
        if end - start >= ctx.config.oscillation_min_length:
            run = list(range(start, end))
            if not set(run) <= covered:
                add('oscillation', run)
                covered.update(run)
            start = end
        else:
            start += 1

    occurrences: Dict[str, List[int]] = {}
    for index, signature in enumerate(signatures):
        occurrences.setdefault(signature, []).append(index)
    for positions in occurrences.values():
        if len(positions) >= ctx.config.repeat_min_count and not set(positions) <= covered:
            add('identical_repeat', positions)

    tail = ctx.config.no_progress_tail
    if len(calls) >= tail:
        seen: Set[str] = set()
        stalled: List[bool] = []
        for index, signature in enumerate(signatures):
            stalled.append(signature in seen or index in ctx.failed_positions)
            seen.add(signature)
        if all(stalled[-tail:]):
            add('no_progress_tail', list(range(len(calls) - tail, len(calls))))
    return findings


def detect_termination(ctx: DetectionContext) -> List[Finding]:
    """Step budget, context overflow, content filter, exhausted nudges, empty answers and unusable scores."""
    view = ctx.view
    findings: List[Finding] = []
    seen: Set[str] = set()
    nudged = any(event.type == 'nudge' for event in view.events)

    def add(subtype: str, confidence: Confidence, detail: str) -> None:
        if subtype in seen:
            return
        seen.add(subtype)
        evidence = [Evidence(excerpt=excerpt(detail, ctx.config.evidence_excerpt_chars))]
        findings.append(
            ctx.finding(FindingCategory.TERMINATION, subtype, TERMINATION, confidence, evidence=evidence, text=detail)
        )

    for event in view.events:
        if event.type == 'error' and event.message in _LOOP_ERROR_SUBTYPES:
            add(_LOOP_ERROR_SUBTYPES[event.message], Confidence.HIGH, event.message)
        elif event.type == 'submit' and event.source == SubmissionSources.PARSE_ERROR_EXHAUSTED:
            add('parse_error_exhausted', Confidence.HIGH, event.source)
        elif event.type == 'submit' and event.source == SubmissionSources.IMPLICIT_NO_NUDGE and nudged:
            # Strategies without nudges (AgentDojo, AgentHarm) end every episode this way; it is only abnormal
            # after the loop actually tried to nudge the model.
            add('nudges_exhausted', Confidence.HIGH, event.source)

    stop_reason = ctx.signals.stop_reason
    if stop_reason in _STOP_REASON_SUBTYPES:
        add(_STOP_REASON_SUBTYPES[stop_reason], Confidence.HIGH, f'stop_reason={stop_reason}')
    if not (view.final_answer or '').strip() and not view.tool_calls and not view.has_nontext_output:
        add('empty_answer', Confidence.MEDIUM, 'empty final answer and no tool call')
    status = (view.score_status or 'success').lower()
    if status not in _USABLE_STATUSES:
        add(f'score_{status}', Confidence.HIGH, f'score status {status}')
    return findings


def _observation_precedes(source: TraceToolCall, call: TraceToolCall) -> bool:
    if source.result_message_index is not None and call.message_index >= 0:
        return source.result_message_index < call.message_index
    if source.result_event_index is None or call.call_event_index is None:
        return False
    # All calls from one generation have their arguments fixed before that step's observations arrive.
    return source.result_event_index < call.call_event_index and source.step != call.step


def _tainted_atoms(
    call: TraceToolCall, ctx: DetectionContext, trusted: str, observations: List[Tuple[TraceToolCall, str, str]]
) -> List[Dict[str, Any]]:
    """Atoms of the call that occur in an earlier tool output but not in the trusted system/user text."""
    tainted: List[Dict[str, Any]] = []
    for kind, atom in extract_atoms(call.arguments, ctx.config.taint_min_string_len):
        if atom in trusted:
            continue
        for source, normalized, raw in observations:
            if not _observation_precedes(source, call):
                continue
            index = normalized.find(atom)
            if index >= 0:
                tainted.append(
                    {
                        'kind': kind,
                        'value': atom[:120],
                        'source_position': source.position,
                        'source_excerpt': excerpt(raw, ctx.config.evidence_excerpt_chars, around=index),
                    }
                )
                break
    return tainted


def _goal_match(expected_args: Dict[str, Any], call: TraceToolCall, trusted: str) -> Optional[Confidence]:
    """Match a goal call on argument values the user never supplied, e.g. the attacker's IBAN or URL."""
    actual = norm(' '.join(iter_strings(call.arguments)))
    identifiers = [
        atom for kind, atom in extract_atoms(expected_args, 10**6) if kind != 'string' and atom not in trusted
    ]
    if identifiers:
        return Confidence.HIGH if any(atom in actual for atom in identifiers) else None
    expected = [
        value for value in (norm(v) for v in iter_strings(expected_args)) if len(value) >= 4 and value not in trusted
    ]
    if not expected:
        return Confidence.MEDIUM
    return Confidence.HIGH if all(value in actual for value in expected) else None


def detect_side_effects(ctx: DetectionContext) -> List[Finding]:
    """Side-effect calls, tainted arguments (provenance), attack-goal calls and risky shell commands."""
    findings: List[Finding] = []
    limit = ctx.config.evidence_excerpt_chars
    verbs = {verb.lower() for verb in ctx.config.side_effect_verbs}
    authoritative = set(ctx.signals.side_effect_tools) if ctx.signals.side_effect_tools is not None else None
    trusted = norm(f'{ctx.view.system_text}\n{ctx.view.user_text}')
    observations = [
        (call, norm(call.result_text), call.result_text) for call in ctx.view.tool_calls if call.result_text
    ]
    goals = list(ctx.signals.attack_goal_calls or [])
    matched_goals: Set[int] = set()

    for call in ctx.view.tool_calls:
        call_evidence = [Evidence(position=call.position, excerpt=excerpt(call_text(call), limit))]
        if is_shell_call(call, ctx.config.shell_tool_names):
            # A plain shell call is not counted as a side effect; only commands matching a risk pattern are.
            command = shell_command(call) or ''
            side_effect = False
            for pattern_id, pattern in ctx.rules.shell_risk.items():
                match = pattern.search(command)
                if match is not None:
                    side_effect = True
                    findings.append(
                        ctx.finding(
                            FindingCategory.RISKY_COMMAND,
                            pattern_id,
                            SIDE_EFFECT,
                            Confidence.MEDIUM,
                            [call.position],
                            [Evidence(position=call.position, excerpt=excerpt(command, limit, match.start()))],
                            {'tool': call.name},
                        )
                    )
        elif authoritative is not None:
            side_effect = call.name in authoritative
        else:
            side_effect = bool(name_tokens(call.name) & verbs)

        if side_effect:
            confidence = Confidence.HIGH if authoritative is not None else Confidence.MEDIUM
            findings.append(
                ctx.finding(
                    FindingCategory.SIDE_EFFECT,
                    'side_effect_call',
                    SIDE_EFFECT,
                    confidence,
                    [call.position],
                    call_evidence,
                    {'tool': call.name},
                )
            )
            tainted = _tainted_atoms(call, ctx, trusted, observations)
            if tainted:
                sources = [
                    Evidence(position=item['source_position'], excerpt=item['source_excerpt']) for item in tainted[:2]
                ]
                atoms = [{k: v for k, v in item.items() if k != 'source_excerpt'} for item in tainted]
                findings.append(
                    ctx.finding(
                        FindingCategory.TAINTED_SIDE_EFFECT,
                        'argument_from_tool_output',
                        SIDE_EFFECT,
                        Confidence.MEDIUM,
                        [call.position] + [item['source_position'] for item in tainted],
                        call_evidence + sources,
                        {'tool': call.name, 'atoms': atoms},
                        first_position=call.position,
                    )
                )
            # Goal ground truths also contain read-only calls shared with the user task; only the
            # state-changing step of a goal is evidence that the attack was carried out.
            for goal_index, goal in enumerate(goals):
                if goal_index in matched_goals or goal.function != call.name:
                    continue
                match_confidence = _goal_match(goal.args, call, trusted)
                if match_confidence is not None:
                    matched_goals.add(goal_index)
                    findings.append(
                        ctx.finding(
                            FindingCategory.ATTACK_GOAL_ACTION,
                            'injection_goal_call',
                            SIDE_EFFECT,
                            match_confidence,
                            [call.position],
                            call_evidence,
                            {'tool': call.name, 'goal_source': goal.source_id, 'expected_args': goal.args},
                        )
                    )
    return findings


def detect_refusal(ctx: DetectionContext, prior: List[Finding]) -> List[Finding]:
    """A provider refusal, or a refusal phrase in the last visible reply of a sample without side effects."""
    last_text = next((text for text in reversed(ctx.view.assistant_texts) if text.strip()), '')
    limit = ctx.config.evidence_excerpt_chars
    outcome = ctx.signals.outcome
    if outcome is not None and outcome.refused is not None:
        if not outcome.refused:
            return []
        evidence = [Evidence(excerpt=excerpt(last_text, limit))] if last_text else []
        return [ctx.finding(FindingCategory.REFUSAL, 'provider_refusal', REFUSAL, Confidence.HIGH, evidence=evidence)]
    if any(f.category in (FindingCategory.SIDE_EFFECT, FindingCategory.RISKY_COMMAND) for f in prior):
        return []
    for pattern in ctx.rules.refusal:
        match = pattern.search(last_text)
        if match is not None:
            evidence = [Evidence(excerpt=excerpt(last_text, limit, match.start()))]
            return [ctx.finding(FindingCategory.REFUSAL, 'refusal_pattern', REFUSAL, Confidence.LOW, evidence=evidence)]
    return []


def run_detectors(ctx: DetectionContext) -> List[Finding]:
    """Run every detector in its fixed order."""
    findings = detect_tool_failures(ctx)
    findings += detect_repetition(ctx)
    findings += detect_termination(ctx)
    findings += detect_side_effects(ctx)
    findings += detect_refusal(ctx, findings)
    return findings


__all__ = [
    'ATOM_PATTERNS',
    'CompiledRules',
    'DetectionContext',
    'call_signature',
    'excerpt',
    'extract_atoms',
    'is_shell_call',
    'iter_strings',
    'name_tokens',
    'norm',
    'run_detectors',
]
