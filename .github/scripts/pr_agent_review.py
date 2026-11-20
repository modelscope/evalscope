"""Run the pinned PR-Agent reviewer with EvalScope's finding schema."""


def review_prompt(template: str) -> str:
    finding_limit = ' (0-{{ num_max_findings }} issues)'
    finding_schema = 'class KeyIssuesComponentLink(BaseModel):\n'
    if finding_limit not in template or finding_schema not in template:
        raise ValueError('The PR-Agent review prompt changed. Update the finding schema override.')
    severity = (
        '    severity: Literal["critical", "high", "medium", "low"] = Field(description="Rate the impact: '
        'critical for severe security exposure, major data loss, or widespread outage; '
        'high for a broken core workflow or systematically wrong evaluation results; '
        'medium for a concrete correctness regression under specific conditions; '
        'low for an actionable defect with limited impact. Do not report cosmetic preferences.")\n'
    )
    return template.replace(finding_limit, '', 1).replace(finding_schema, finding_schema + severity, 1)


def main() -> None:
    from pr_agent.cli import run
    from pr_agent.config_loader import get_settings

    settings = get_settings()
    settings.set('PR_REVIEW_PROMPT.SYSTEM', review_prompt(settings.pr_review_prompt.system))
    raise SystemExit(run())


if __name__ == '__main__':
    main()
