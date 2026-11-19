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


def test_render_only_changed_files_and_suppresses_mentions() -> None:
    finding = {
        'relevant_file': 'a.py',
        'start_line': 12,
        'issue_header': 'Header',
        'issue_content': '<!-- evasion --> @someone [text](https://example.org)',
    }
    result = {'review': {'key_issues_to_review': [finding, dict(finding, relevant_file='unrelated.py')]}}
    body = bot.render(result, ['a.py'])
    assert 'a.py:12' in body
    assert 'unrelated.py' not in body
    assert '<!--' not in body
    assert '@someone' not in body


def test_failed_review_updates_comment_and_fails_the_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / 'request.json').write_text(json.dumps({'actor': 'maintainer'}))
    monkeypatch.setenv('REVIEW_RESULT', 'failed')
    monkeypatch.setattr(bot, 'current_pr', lambda *_: PR)
    previous = '## EvalScope AI review\n\nPrevious review'
    monkeypatch.setattr(bot, 'previous_review', lambda _: (17, {'reviewed': True}, previous))
    calls = []

    def github(path: str, *args: str, body: dict | None = None) -> None:
        calls.append((path, args, body))

    monkeypatch.setattr(bot, 'github', github)
    with pytest.raises(SystemExit, match='did not return a review'):
        bot.publish(42, SHA, 23, tmp_path)
    assert calls == [
        (
            'issues/comments/17',
            ('-X', 'PATCH'),
            {'body': f'<!-- evalscope-pr-review {{"reviewed": true, "last_command": 23}} -->\n{previous}'},
        )
    ]
