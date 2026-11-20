import json
from pathlib import Path
from urllib.parse import urlsplit

import pr_automation as bot
import pytest
import yaml

SHA = 'a' * 40
PR = {
    'number': 42,
    'state': 'open',
    'draft': False,
    'base': {'ref': 'main'},
    'head': {'sha': SHA, 'ref': 'updates', 'repo': {'id': 123}},
    'labels': [],
    'user': {'id': 1},
}
RUN = {
    'id': 10,
    'event': 'pull_request',
    'conclusion': 'success',
    'status': 'completed',
    'head_repository': {'id': 123, 'owner': {'login': 'contributor'}},
    'head_branch': 'updates',
    'head_sha': SHA,
    'pull_requests': [],
}


def test_resolve_fork_without_pr_number(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    event = tmp_path / 'event.json'
    event.write_text(json.dumps({'workflow_run': dict(RUN, path=bot.INTAKE)}))
    output = tmp_path / 'output'
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'workflow_run')
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(event))
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    monkeypatch.setattr(bot, 'get', lambda _: [dict(PR, head=dict(PR['head'], sha='b' * 40))])
    bot.resolve()
    assert not output.exists()

    monkeypatch.setattr(bot, 'get', lambda _: [PR])
    bot.resolve()
    assert output.read_text().splitlines() == ['number=42', f'sha={SHA}', 'command=0', 'intake=true']


def test_ci_waits_for_frontend_and_ignores_push_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {
        'ci_test.yaml': [RUN],
        'lint.yml': [RUN],
        'frontend.yml': [dict(RUN, event='push')],
    }

    def get_runs(url: str) -> dict:
        workflow = urlsplit(url).path.split('/')[-2]
        return {'workflow_runs': runs[workflow]}

    monkeypatch.setattr(bot, 'get', get_runs)
    assert bot.ci_passed(PR, ['evalscope/benchmarks/example.py'])
    assert not bot.ci_passed(PR, ['evalscope/web/src/App.tsx'])
    runs['frontend.yml'].append(RUN)
    assert bot.ci_passed(PR, ['evalscope/web/src/App.tsx'])
    runs['lint.yml'].append(dict(RUN, id=11, conclusion='failure'))
    assert not bot.ci_passed(PR, ['evalscope/web/src/App.tsx'])


def test_frontend_paths_match_workflow() -> None:
    workflow = Path(__file__).resolve().parent.parent / 'workflows' / 'frontend.yml'
    content = yaml.safe_load(workflow.read_text())
    assert bot.FRONTEND_PATHS == tuple(content[True]['pull_request']['paths'])


@pytest.mark.parametrize(
    ('permission', 'user_id', 'actor'),
    [('write', 2, 'maintainer'), ('read', 1, 'author'), ('read', 2, '')],
)
def test_command_actor(monkeypatch: pytest.MonkeyPatch, permission: str, user_id: int, actor: str) -> None:
    records = {
        'issues/comments/15': {
            'issue_url': '/issues/42',
            'body': '/review',
            'user': {
                'id': user_id,
                'login': 'person',
                'type': 'User',
            },
        },
        'collaborators/person/permission': {'permission': permission},
    }
    monkeypatch.setattr(bot, 'get', records.__getitem__)
    assert bot.command_actor(42, 15, dict(PR, user={'id': 1})) == actor


def test_skip_reason_explains_the_actual_diff_limit() -> None:
    assert bot.skip_reason(15, [], b'x' * 62068) == (
        'The PR changes 15 files and the raw diff is 62,068 bytes (60.6 KiB). The diff limit is 49,152 bytes (48 KiB).'
    )
    assert bot.skip_reason(51, [], None) == 'The PR changes 51 files. The review limit is 50.'
    assert bot.skip_reason(1, [{}], b'x') == 'GitHub did not supply a text patch for 1 of 1 changed files.'
    assert bot.skip_reason(1, [{'patch': 'x'}], b'x' * bot.MAX_DIFF_BYTES) == ''


def test_failed_review_updates_comment_and_fails_the_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / 'request.json').write_text(json.dumps({'actor': 'maintainer'}))
    monkeypatch.setenv('REVIEW_RESULT', 'failed')
    monkeypatch.setenv('GITHUB_SERVER_URL', 'https://github.com')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_RUN_ID', '918')
    monkeypatch.setattr(bot, 'current_pr', lambda *_: dict(PR, labels=[{'name': 'ai:findings'}]))
    previous = '## EvalScope AI review\n\nPrevious review'
    monkeypatch.setattr(bot, 'previous_review', lambda _: (17, {'reviewed': True}, previous))
    calls = []

    def github(path: str, *args: str, body: dict | None = None) -> None:
        calls.append((path, args, body))

    monkeypatch.setattr(bot, 'github', github)
    with pytest.raises(SystemExit, match='did not return a review'):
        bot.publish(42, SHA, 23, tmp_path)
    path, args, payload = calls[0]
    assert (path, args) == ('issues/comments/17', ('-X', 'PATCH'))
    body = payload['body']
    assert '"last_command": 23' in body
    assert f'"result_sha": "{SHA}"' in body
    assert '**Review failed**' in body
    assert '[Actions run](https://github.com/owner/repo/actions/runs/918)' in body
    assert bot.PREVIOUS_REVIEW + 'Previous review' in body
    assert calls[1:] == [
        ('issues/42/labels', ('-X', 'POST'), {'labels': ['ai:failed']}),
        ('issues/42/labels/ai%3Afindings', ('-X', 'DELETE'), None),
    ]


@pytest.mark.parametrize(
    ('outcome', 'label'),
    [('ai:no-findings', ''), ('ai:skipped', ''), ('ai:findings', 'ai:findings'), ('ai:failed', 'ai:failed')],
)
def test_review_label_only_restores_alerts(outcome: str, label: str) -> None:
    assert bot.review_label({'result_sha': SHA, 'outcome': outcome}, '', SHA) == label


@pytest.mark.parametrize(
    ('result', 'title'),
    [('dsh-success', 'No actionable findings'), ('size-skip', 'Review skipped')],
)
def test_quiet_review_updates_comment_and_clears_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, result: str, title: str
) -> None:
    (tmp_path / 'request.json').write_text(json.dumps({'files': [], 'skip_reason': 'Diff too large.'}))
    write_dsh_summary(tmp_path)
    monkeypatch.setenv('REVIEW_RESULT', dsh_result(0) if result == 'dsh-success' else result)
    monkeypatch.setenv('GITHUB_SERVER_URL', 'https://github.com')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_RUN_ID', '918')
    labels = [{'name': name} for name in ('ai:findings', 'ai:no-findings', 'area:core')]
    monkeypatch.setattr(bot, 'current_pr', lambda *_: dict(PR, labels=labels))
    monkeypatch.setattr(bot, 'previous_review', lambda _: (17, {}, ''))
    calls = []

    def github(path: str, *args: str, body: dict | None = None) -> None:
        calls.append((path, args, body))

    monkeypatch.setattr(bot, 'github', github)
    bot.publish(42, SHA, 0, tmp_path)
    path, args, payload = calls[0]
    assert (path, args) == ('issues/comments/17', ('-X', 'PATCH'))
    assert f'**{title}**' in payload['body']
    assert calls[1:] == [
        ('issues/42/labels/ai%3Afindings', ('-X', 'DELETE'), None),
        ('issues/42/labels/ai%3Ano-findings', ('-X', 'DELETE'), None),
    ]


def test_triage_pr_expires_status_for_an_old_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = ['area:core', 'needs-info', 'ai:findings']
    monkeypatch.setattr(bot, 'get', lambda _: dict(PR, labels=[{'name': label} for label in labels]))
    monkeypatch.setattr(bot, 'previous_review', lambda _: (17, {'result_sha': 'b' * 40, 'outcome': 'ai:findings'}, ''))
    monkeypatch.setenv('PR_REQUEUE', 'true')
    calls = []

    def github(path: str, *args: str, body: dict | None = None) -> None:
        calls.append((path, args, body))

    monkeypatch.setattr(bot, 'github', github)
    bot.triage_pr(42, SHA)
    assert calls == [
        ('issues/42/labels/needs-info', ('-X', 'DELETE'), None),
        ('issues/42/labels/ai%3Afindings', ('-X', 'DELETE'), None),
    ]


def test_triage_resolves_conflicting_states_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    item = {
        'number': 42,
        'state': 'open',
        'labels': [{'name': 'needs-info'}, {'name': 'area:core'}, {'name': 'accepted'}],
    }
    events = [
        {'event': 'labeled', 'label': {'name': 'accepted'}, 'created_at': '2026-11-19T08:10:00Z', 'id': 2},
        {'event': 'labeled', 'label': {'name': 'needs-info'}, 'created_at': '2026-11-19T08:09:00Z', 'id': 1},
    ]
    monkeypatch.setattr(bot, 'list_all', lambda _: events)
    calls = []

    def github(path: str, *args: str, body: dict | None = None) -> None:
        calls.append((path, args, body))

    monkeypatch.setattr(bot, 'github', github)
    bot.sync_triage(item)
    assert calls == [('issues/42/labels/needs-info', ('-X', 'DELETE'), None)]


@pytest.mark.parametrize('item', [PR, {'number': 43, 'state': 'open', 'labels': [{'name': 'area:core'}]}])
def test_new_open_items_without_a_triage_status_keep_their_labels(monkeypatch: pytest.MonkeyPatch, item: dict) -> None:
    def github(*args: object, **kwargs: object) -> None:
        pytest.fail('Triage should not update labels when there is no triage status.')

    monkeypatch.setattr(bot, 'github', github)
    bot.sync_triage(item)


def write_dsh_summary(directory: Path) -> None:
    (directory / 'dsh-summary.md').write_text(
        '<!-- dsh-action:v1 kind=summary -->\n## DeepSeek Harness review\n\n### Findings carried in the summary\n'
    )


def dsh_result(selected: int, status: str = 'success') -> str:
    return json.dumps(
        {
            'schemaVersion': 1,
            'status': status,
            'operation': 'review',
            'publication': {'selected': selected},
        }
    )


def test_dsh_summary_preserves_findings_without_an_inline_location(tmp_path: Path) -> None:
    write_dsh_summary(tmp_path)
    assert bot.review_result(dsh_result(1), {}, tmp_path) == ('ai:findings', '### Findings carried in the summary')
    assert bot.review_result(dsh_result(1, 'failed'), {}, tmp_path)[0] == 'ai:failed'
    (tmp_path / 'dsh-summary.md').unlink()
    assert bot.review_result(dsh_result(1), {}, tmp_path)[0] == 'ai:failed'


def test_prepare_binds_dsh_review_to_the_checked_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repository = {'id': 5, 'name': 'repo', 'full_name': 'owner/repo', 'owner': {'login': 'owner'}}
    pr = dict(PR, base=dict(PR['base'], repo=repository), changed_files=1)
    output = tmp_path / 'output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    monkeypatch.setattr(bot, 'previous_review', lambda _: (None, {}, ''))
    monkeypatch.setattr(bot, 'command_actor', lambda *_: 'maintainer')
    files = [{'filename': 'evalscope/benchmarks/example/example_adapter.py', 'status': 'added', 'patch': '+change'}]
    monkeypatch.setattr(bot, 'list_all', lambda _: files)
    monkeypatch.setattr(bot, 'github', lambda *_: b'+change')
    monkeypatch.setattr(bot, 'get', lambda _: pr)

    bot.prepare(42, SHA, 2, tmp_path)
    event = json.loads((tmp_path / 'pull-request.json').read_text())
    assert event['repository'] == repository and event['pull_request']['head']['sha'] == SHA
    assert output.read_text() == 'decision=review\n'
    prompt = (tmp_path / 'prompt.txt').read_text()
    assert 'critical' in prompt and 'def format_prompt_template(' in prompt
    assert 'Write all user-facing review text in English' in prompt
    assert 'verified required CI' not in prompt

    moved = dict(pr, head=dict(pr['head'], sha='b' * 40))
    snapshots = iter((pr, moved))
    monkeypatch.setattr(bot, 'get', lambda _: next(snapshots))
    not_prepared = tmp_path / 'stale'
    bot.prepare(42, SHA, 2, not_prepared)
    assert not not_prepared.exists()
