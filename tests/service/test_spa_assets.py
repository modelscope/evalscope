"""Missing versioned assets must not masquerade as successful SPA navigation."""

from pathlib import Path

import pytest

from evalscope.service import app as service_app


def test_missing_assets_are_404_and_navigation_html_is_revalidated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'index.html').write_text('<html>EvalScope entry</html>')
    (tmp_path / 'assets' / 'current.js').write_text('export const loaded = true;')
    monkeypatch.setattr(service_app, '_WEB_DIST', str(tmp_path))
    client = service_app.create_app().test_client()
    for path in ('/assets/old-math.js', '/assets/old-math.css'):
        response = client.get(path)
        assert response.status_code == 404
        assert b'EvalScope entry' not in response.data
    response = client.get('/assets/current.js')
    assert response.status_code == 200 and response.mimetype != 'text/html'
    assert b'export const loaded' in response.data
    response = client.get('/reports/run/model')
    assert response.status_code == 200 and b'EvalScope entry' in response.data
    assert response.headers['Cache-Control'] == 'no-cache'
