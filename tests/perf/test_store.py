import json
import sqlite3

from evalscope.perf.domain.observation import RequestObservation
from evalscope.perf.results import SQLiteResultStore


def test_sqlite_store_uses_versioned_json_payload(tmp_path) -> None:
    path = tmp_path / 'observations.sqlite'
    store = SQLiteResultStore(str(path), commit_interval=1)
    store.open()
    store.write(
        RequestObservation(
            run_id='run',
            request_id='request',
            success=True,
            response_payloads=[{
                'text': '你好'
            }],
        )
    )
    observations = list(store.observations())
    store.close()
    assert observations[0].response_payloads == [{'text': '你好'}]
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT version FROM schema_info').fetchone() == (1, )
        payload = connection.execute('SELECT payload FROM observations').fetchone()[0]
    assert json.loads(payload)['response_payloads'] == [{'text': '你好'}]
