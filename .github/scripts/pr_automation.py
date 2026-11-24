"""Route PR events and publish an advisory review for a verified head commit."""

import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from textwrap import shorten
from typing import Any
from urllib.parse import quote, urlencode

INTAKE = '.github/workflows/pr-intake.yml'
CI = ('ci_test.yaml', 'lint.yml')
FRONTEND = 'frontend.yml'
# Keep these paths in sync with the pull_request filter in frontend.yml.
FRONTEND_PATHS = (
    'evalscope/web/**',
    'evalscope/service/**',
    'evalscope/report/report.py',
    'evalscope/api/agent/trace.py',
    'evalscope/api/metric/**',
    'evalscope/constants.py',
    'scripts/generate_web_api_contracts.py',
    'tests/service/test_api_contracts.py',
    'tests/report/test_report_endpoints.py',
    'requirements/dev.txt',
    'requirements/service.txt',
    'pyproject.toml',
    'DESIGN.md',
    '.github/workflows/frontend.yml',
)
MAX_FILES = 50
MAX_DIFF_BYTES = 256 * 1024
MAX_COMMENT_CHARACTERS = 65_536
TRIAGE_LABELS = frozenset(('needs-info', 'accepted', 'ready-for-review'))
AI_TITLES = {
    'ai:no-findings': 'No actionable findings',
    'ai:limited': 'Limited review',
    'ai:findings': 'Findings to verify',
    'ai:failed': 'Review failed',
    'ai:skipped': 'Review skipped',
}
AI_ALERT_LABELS = frozenset(('ai:findings', 'ai:failed'))
# Include retired labels so synchronization removes them from existing PRs.
AI_LABELS = frozenset(AI_TITLES)
AI_SUCCESS = frozenset(('ai:no-findings', 'ai:findings', 'ai:limited'))
PREVIOUS_REVIEW = '\n\n### Previous successful review\n\n'
STALE_MARKER = '<!-- evalscope-pr-review-outdated -->'
MARKER = re.compile(r'<!-- evalscope-pr-review (\{[^\n]*\}) -->\n')
LEGACY_COMMIT = re.compile(r'^Commit ([0-9a-f]{12})$', re.MULTILINE)
SHA = re.compile(r'[0-9a-f]{40}')
SEVERITIES = ('critical', 'high', 'medium')
CI_NAMES = {'ci_test.yaml': 'CI Lite', 'lint.yml': 'Lint', 'frontend.yml': 'Frontend'}


def github(path: str, *args: str, body: dict | None = None) -> bytes:
    command = ['gh', 'api', f'repos/{os.environ["GITHUB_REPOSITORY"]}/{path}', *args]
    payload = None
    if body is not None:
        command += ['--input', '-']
        payload = json.dumps(body).encode()
    return subprocess.run(command, input=payload, capture_output=True, check=True).stdout


def query(path: str, **params: str) -> str:
    return f'{path}?{urlencode(params)}'


def get(path: str) -> Any:
    return json.loads(github(path))


def output(**values: Any) -> None:
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as handle:
        for name, value in values.items():
            handle.write(f'{name}={value}\n')


def read_event() -> dict:
    return json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())


def targets_main(pr: dict) -> bool:
    return pr['state'] == 'open' and pr['base']['ref'] == 'main'


def run_matches(run: dict, pr: dict) -> bool:
    head = pr['head']
    return (
        run['event'] == 'pull_request'
        and head['repo'] is not None
        and (run['head_repository'] or {}).get('id') == head['repo']['id']
        and run['head_branch'] == head['ref']
        and run['head_sha'] == head['sha']
    )


def resolve() -> None:
    event = read_event()
    if os.environ['GITHUB_EVENT_NAME'] == 'issue_comment':
        comment = event.get('comment', {})
        if 'pull_request' not in event.get('issue', {}) or comment.get('body', '').strip() != '/review':
            return
        pr = get(f'pulls/{int(event["issue"]["number"])}')
        if targets_main(pr) and not pr['draft']:
            output(number=pr['number'], sha=pr['head']['sha'], command=comment['id'], intake='false')
        return

    run = event['workflow_run']
    repo = run['head_repository']
    if run['event'] != 'pull_request' or run['conclusion'] != 'success' or repo is None:
        return
    head = f'{repo["owner"]["login"]}:{run["head_branch"]}'
    candidates = get(query('pulls', state='open', base='main', head=head))
    matches = [pr for pr in candidates if targets_main(pr) and run_matches(run, pr)]
    if len(matches) == 1:
        pr = matches[0]
        intake = str(run['path'] == INTAKE).lower()
        output(number=pr['number'], sha=pr['head']['sha'], command=0, intake=intake)
        if (
            intake == 'true'
            and run.get('display_title') in ('PR Intake (synchronize)', 'PR Intake (edited)')
            and (run.get('actor') or {}).get('id') == pr['user']['id']
        ):
            output(requeue='true')


def ci_runs(pr: dict, files: list[str]) -> dict[str, dict]:
    workflows = list(CI)
    if any(fnmatchcase(file, pattern) for file in files for pattern in FRONTEND_PATHS):
        workflows.append(FRONTEND)
    latest = {}
    for workflow in workflows:
        path = query(f'actions/workflows/{workflow}/runs', event='pull_request', head_sha=pr['head']['sha'])
        runs = (run for run in get(path)['workflow_runs'] if run_matches(run, pr))
        latest[workflow] = max(runs, key=lambda run: run['id'], default={})
    return latest


def ci_passed(runs: dict[str, dict]) -> bool:
    return all(run.get('status') == 'completed' and run.get('conclusion') == 'success' for run in runs.values())


def ci_evidence(sha: str, runs: dict[str, dict]) -> dict:
    workflows = []
    for name, run in runs.items():
        if not run:
            workflows.append({'workflow': name, 'status': 'unavailable'})
            continue
        record = {key: run[key] for key in ('id', 'status', 'conclusion', 'html_url', 'run_attempt')}
        path = f'actions/runs/{run["id"]}/attempts/{run["run_attempt"]}/jobs'
        jobs = []
        for page in json.loads(github(f'{path}?per_page=100', '--paginate', '--slurp')):
            for job in page['jobs']:
                item = {key: job[key] for key in ('name', 'status', 'conclusion', 'html_url')}
                item['steps'] = [
                    {key: step[key] for key in ('name', 'status', 'conclusion')} for step in job.get('steps', [])
                ]
                jobs.append(item)
        workflows.append({'workflow': name, **record, 'jobs': jobs})
    return {
        'head_sha': sha,
        'required_ci_passed': ci_passed(runs),
        'scope': 'GitHub job and step status. Step names do not establish individual test cases executed.',
        'workflows': workflows,
    }


def list_all(path: str) -> list[dict]:
    pages = json.loads(github(f'{path}?per_page=100', '--paginate', '--slurp'))
    return [item for page in pages for item in page]


def previous_review(number: int) -> tuple[int | None, dict, str]:
    for comment in list_all(f'issues/{number}/comments'):
        if comment['user']['login'] == 'github-actions[bot]' and (match := MARKER.match(comment['body'])):
            return comment['id'], json.loads(match[1]), comment['body'][match.end() :]
    return None, {}, ''


def replace_family(number: int, labels: set[str], family: set[str] | frozenset[str], keep: str = '') -> None:
    if keep and keep not in labels:
        github(f'issues/{number}/labels', '-X', 'POST', body={'labels': [keep]})
    for label in sorted((labels & family) - {keep}):
        try:
            github(f'issues/{number}/labels/{quote(label, safe="")}', '-X', 'DELETE')
        except subprocess.CalledProcessError as error:
            if b'HTTP 404' not in error.stderr:
                raise


def latest_status(number: int, states: set[str]) -> str:
    if len(states) == 1:
        return next(iter(states))
    events = (
        event
        for event in list_all(f'issues/{number}/events')
        if event['event'] == 'labeled' and (event.get('label') or {}).get('name') in states
    )
    latest = max(events, key=lambda event: (event['created_at'], event['id']), default=None)
    return latest['label']['name'] if latest else 'needs-info'


def sync_triage(item: dict, *, requeue: bool = False) -> None:
    if item['state'] != 'open':
        return
    is_pr = 'pull_request' in item or 'draft' in item
    labels = {label['name'] for label in item['labels']}
    allowed = TRIAGE_LABELS - ({'accepted'} if is_pr else {'ready-for-review'})
    states = labels & allowed
    status = latest_status(item['number'], states) if states else ''
    if requeue and status == 'needs-info':
        status = ''
    replace_family(item['number'], labels, TRIAGE_LABELS, status)


def triage_event() -> None:
    event = read_event()
    snapshot = event['issue']
    item = get(f'issues/{snapshot["number"]}')
    if os.environ['GITHUB_EVENT_NAME'] == 'issue_comment':
        comment = event['comment']
        if (
            comment['user']['id'] == item['user']['id']
            and not comment['body'].lstrip().startswith('/')
            and any(label['name'] == 'needs-info' for label in item['labels'])
        ):
            sync_triage(item, requeue=True)
        return
    requeue = event['action'] == 'edited' and event['sender']['id'] == item['user']['id']
    sync_triage(item, requeue=requeue)


def review_label(state: dict, previous: str, sha: str) -> str:
    if 'result_sha' in state:
        if state['result_sha'] == sha and state.get('outcome') in AI_ALERT_LABELS:
            return state['outcome']
        return ''
    # Summaries published before review labels used a standalone Commit line.
    if (legacy := LEGACY_COMMIT.search(previous)) is None or legacy[1] != sha[:12]:
        return ''
    if state.get('size_skip_sha') == sha and 'Review skipped' in previous:
        return ''
    if 'The model did not return a review.' in previous or 'could be located' in previous:
        return 'ai:failed'
    if state.get('reviewed') and re.search(r'^- \*\*', previous, re.MULTILINE):
        return 'ai:findings'
    return ''


def triage_pr(number: int, sha: str) -> None:
    pr = get(f'pulls/{number}')
    if not targets_main(pr) or pr['head']['sha'] != sha:
        return
    sync_triage(pr, requeue=os.environ.get('PR_REQUEUE') == 'true')
    comment_id, state, previous = previous_review(number)
    labels = {label['name'] for label in pr['labels']}
    replace_family(number, labels, AI_LABELS, review_label(state, previous, sha))
    reviewed = state.get('result_sha')
    if not isinstance(reviewed, str) or not SHA.fullmatch(reviewed):
        legacy = LEGACY_COMMIT.search(previous)
        reviewed = legacy[1] if legacy else ''
    if comment_id is not None and previous and reviewed and not sha.startswith(reviewed):
        _, separator, historical = previous.partition(f'{STALE_MARKER}\n\n')
        if not separator:
            historical = previous
        repo = f'{os.environ["GITHUB_SERVER_URL"]}/{os.environ["GITHUB_REPOSITORY"]}/pull/{number}/commits'
        notice = (
            f'> **Review outdated** · Reviewed [`{reviewed[:12]}`]({repo}/{reviewed})'
            f' · Current [`{sha[:12]}`]({repo}/{sha})\n\n{STALE_MARKER}\n\n'
        )
        if notice + historical != previous and current_pr(number, sha) is not None:
            body = f'<!-- evalscope-pr-review {json.dumps(state)} -->\n' + notice + historical
            github(f'issues/comments/{comment_id}', '-X', 'PATCH', body={'body': body[:MAX_COMMENT_CHARACTERS]})


def current_pr(number: int, sha: str) -> dict | None:
    pr = get(f'pulls/{number}')
    if targets_main(pr) and not pr['draft'] and pr['head']['sha'] == sha:
        return pr
    print('The PR is no longer open and ready at the selected commit.')
    return None


def command_actor(number: int, command: int, pr: dict) -> str:
    comment = get(f'issues/comments/{command}')
    if not comment['issue_url'].endswith(f'/issues/{number}') or comment['body'].strip() != '/review':
        return ''
    user = comment['user']
    if user['type'] != 'User':
        return ''
    permission = get(f'collaborators/{user["login"]}/permission')['permission']
    if permission in ('admin', 'write'):
        return 'maintainer'
    return 'author' if user['id'] == pr['user']['id'] else ''


def skip_reason(count: int, files: list[dict], diff: bytes | None) -> str:
    if count > MAX_FILES:
        return f'The PR changes {count} files. The review limit is {MAX_FILES}.'
    if diff is None:
        return ''
    if len(diff) > MAX_DIFF_BYTES:
        return (
            f'The PR changes {count} files and the raw diff is {len(diff):,} bytes ({len(diff) / 1024:.1f} KiB). '
            f'The diff limit is {MAX_DIFF_BYTES:,} bytes ({MAX_DIFF_BYTES // 1024} KiB).'
        )
    missing = sum('patch' not in file for file in files)
    if missing:
        return f'GitHub did not supply a text patch for {missing} of {count} changed files.'
    if not diff.strip():
        return 'GitHub supplied an empty diff.'
    return ''


def prepare(number: int, sha: str, command: int, directory: Path) -> None:
    pr = current_pr(number, sha)
    if pr is None:
        return
    _, state, _ = previous_review(number)
    if command:
        actor = command_actor(number, command, pr)
        if not actor or command <= state.get('last_command', 0):
            return
        if actor == 'author' and sha in state.get('author_shas', []):
            return
    else:
        actor = 'automatic'
        start = os.environ.get('PR_REVIEW_AUTO_START_AT', '')
        if not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', start) or pr['created_at'] < start:
            return
        newer_with_findings = state.get('outcome') == 'ai:findings' and state.get('result_sha') != sha
        if (state.get('reviewed') and not newer_with_findings) or state.get('size_skip_sha') == sha:
            return

    count = pr['changed_files']
    files = list_all(f'pulls/{number}/files')
    paths = [file['filename'] for file in files] + [
        file['previous_filename'] for file in files if 'previous_filename' in file
    ]
    runs = ci_runs(pr, paths)
    if actor != 'maintainer' and (len(files) != count or not ci_passed(runs)):
        return

    base_sha = pr['base']['sha']
    diff = github(f'pulls/{number}', '-H', 'Accept: application/vnd.github.diff') if count <= MAX_FILES else None
    pr = current_pr(number, sha)
    if pr is None or pr['base']['sha'] != base_sha:
        return

    reason = skip_reason(count, files, diff)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'request.json').write_text(json.dumps({'actor': actor, 'skip_reason': reason}))
    if reason:
        output(decision='size-skip')
        return
    event = {'repository': pr['base']['repo'], 'action': 'synchronize', 'pull_request': pr}
    (directory / 'pull-request.json').write_text(json.dumps(event))
    (directory / 'prompt.txt').write_text(Path(__file__).with_name('dsh_review_prompt.md').read_text())
    (directory / 'changes.diff').write_bytes(diff)
    (directory / 'ci.json').write_text(json.dumps(ci_evidence(sha, runs), ensure_ascii=False, indent=2))
    output(decision='review')


def review_result(result: str, request: dict, directory: Path) -> tuple[str, str]:
    if result == 'size-skip':
        return 'ai:skipped', request['skip_reason']
    try:
        data = json.loads(result)
        publication = data['publication']
        selected = publication['selected']
        summary_only = publication['summaryOnly']
        omitted = publication.get('summaryOmitted', 0)
        summary = (directory / 'dsh-summary.md').read_text()
        findings = json.loads((directory / 'dsh-findings.json').read_text())
        evidence = json.loads((directory / 'ci.json').read_text())
        candidates = findings['candidates']
        severities = findings['severities']
    except (OSError, ValueError, TypeError, KeyError):
        return 'ai:failed', 'DSH-action did not return a usable review. See the Actions run for details.'
    if (
        data.get('schemaVersion') != 1
        or data.get('status') != 'success'
        or data.get('operation') != 'review'
        or not summary.startswith('<!-- dsh-action:v1 kind=summary -->\n')
        or not all(type(count) is int and count >= 0 for count in (candidates, selected, summary_only, omitted))
        or not summary_only + omitted <= selected <= candidates
        or not isinstance(severities, list)
        or len(severities) != selected
        or any(severity not in SEVERITIES for severity in severities)
    ):
        return 'ai:failed', 'DSH-action did not return a usable review. See the Actions run for details.'

    checks = []
    for workflow in evidence.get('workflows', []):
        name = CI_NAMES[workflow['workflow']]
        status = workflow.get('conclusion') or workflow['status']
        text = f'{name} {"passed" if status == "success" else status.replace("_", " ")}'
        url = workflow.get('html_url')
        checks.append(f'[{text}]({url})' if url else text)
    lines = ['Static review' + (' · ' + ' · '.join(checks) if checks else '') + '.']
    overview, _, details = summary.partition('\n\n### Findings carried in the summary\n\n')
    # DSH's rendered form has already stripped tracking markers and redacted secrets.
    limitation = next((line for line in overview.splitlines() if line.startswith('Limited: ')), '')
    if limitation:
        lines.append(html.escape(limitation[:200]))
    elif candidates == 0:
        checked = next(
            (line.removeprefix('Checked: ').strip() for line in overview.splitlines() if line.startswith('Checked: ')),
            '',
        )
        if checked:
            note = shorten(checked, width=300, placeholder='…')
            lines.append(f'**What was checked** · {html.escape(note, quote=False)}')
    if selected:
        ratings = ', '.join(f'{severities.count(level)} {level}' for level in SEVERITIES if level in severities)
        lines.append(f'**{ratings}** · {selected - summary_only - omitted} inline · {summary_only} in summary.')
    if candidates > selected:
        lines.append(f'{candidates - selected} candidate(s) did not meet publication criteria.')
    if omitted:
        lines.append(f'{omitted} finding(s) exceeded GitHub’s summary limit.')
    if details:
        lines.append(
            f'<details>\n<summary>Findings without an inline comment</summary>\n\n{details.strip()}\n\n</details>'
        )
    if notes := findings.get('notes'):
        rendered_notes = []
        for note in notes[:5]:
            title = html.escape(' '.join(note['title'].split())[:160])
            path = html.escape(note['path'][:240])
            proof = html.escape(' '.join(note['evidence'].split())[:400])
            rendered_notes.append(f'- **{title}** · `{path}:{note["line"]}` · {note["confidence"]:.0%}\n  {proof}')
        if len(notes) > len(rendered_notes):
            rendered_notes.append(f'Showing {len(rendered_notes)} of {len(notes)} candidates with evidence.')
        rendered = '\n\n'.join(rendered_notes)
        lines.append(
            f'<details>\n<summary>Candidate evidence for human verification</summary>\n\n{rendered}\n\n</details>'
        )
    outcome = 'ai:no-findings'
    if selected:
        outcome = 'ai:findings'
    elif limitation or candidates:
        outcome = 'ai:limited'
    return outcome, '\n\n'.join(lines)


def summary_body(number: int, sha: str, outcome: str, text: str) -> str:
    repo = f'{os.environ["GITHUB_SERVER_URL"]}/{os.environ["GITHUB_REPOSITORY"]}'
    commit = f'[{sha[:12]}]({repo}/pull/{number}/commits/{sha})'
    run = f'{repo}/actions/runs/{os.environ["GITHUB_RUN_ID"]}'
    completed = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    status = f'**{AI_TITLES[outcome]}** · {commit} · {completed} · [Actions run]({run})'
    return f'## EvalScope AI review\n\n{status}\n\n{text}\n'


def publish(number: int, sha: str, command: int, directory: Path) -> None:
    if current_pr(number, sha) is None:
        return
    request = json.loads((directory / 'request.json').read_text())
    comment_id, state, previous = previous_review(number)
    outcome, text = review_result(os.environ['REVIEW_RESULT'], request, directory)
    body = summary_body(number, sha, outcome, text)
    if outcome == 'ai:failed':
        prior = previous.partition(PREVIOUS_REVIEW)[2]
        previous_outcome = state.get('outcome')
        if previous_outcome in AI_SUCCESS or ('outcome' not in state and state.get('reviewed')):
            prior = previous.removeprefix('## EvalScope AI review\n\n')
        if prior:
            body = body.rstrip() + PREVIOUS_REVIEW + prior.rstrip() + '\n'
    if command:
        state['last_command'] = command
        if request['actor'] == 'author':
            state.setdefault('author_shas', []).append(sha)
    if outcome in AI_SUCCESS:
        state['reviewed'] = True
    if outcome == 'ai:skipped':
        state['size_skip_sha'] = sha
    state.update(result_sha=sha, outcome=outcome)
    pr = current_pr(number, sha)
    if pr is None:
        return
    metadata = f'<!-- evalscope-pr-review {json.dumps(state)} -->\n'
    if len(metadata) + len(body) > MAX_COMMENT_CHARACTERS:
        notice = '\n\n[Review summary truncated to fit the GitHub comment limit.]\n'
        body = body[: MAX_COMMENT_CHARACTERS - len(metadata) - len(notice)].rstrip() + notice
    comment = {'body': metadata + body}
    if comment_id is None:
        github(f'issues/{number}/comments', '-X', 'POST', body=comment)
    else:
        github(f'issues/comments/{comment_id}', '-X', 'PATCH', body=comment)
    label = outcome if outcome in AI_ALERT_LABELS else ''
    replace_family(number, {label['name'] for label in pr['labels']}, AI_LABELS, label)
    if outcome == 'ai:failed':
        raise SystemExit('DSH-action did not return a review.')


def main() -> None:
    phase = sys.argv[1]
    if phase == 'resolve':
        resolve()
        return
    if phase == 'triage-event':
        triage_event()
        return
    directory = Path(os.environ['RUNNER_TEMP']) / 'evalscope-review'
    number = int(os.environ['PR_NUMBER'])
    sha = os.environ['PR_SHA']
    if not SHA.fullmatch(sha):
        raise ValueError('Unexpected PR head SHA.')
    if phase == 'triage-pr':
        triage_pr(number, sha)
        return
    command = int(os.environ['PR_COMMAND'])
    if phase == 'prepare':
        prepare(number, sha, command, directory)
    elif phase == 'publish':
        publish(number, sha, command, directory)


if __name__ == '__main__':
    main()
