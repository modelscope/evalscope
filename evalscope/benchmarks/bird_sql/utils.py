# Execution-based scoring helpers for BIRD (text-to-SQL on SQLite).
#
# The schema prompt follows the reference code in https://github.com/bird-bench/mini_dev
# (llm/src/table_schema.py, llm/src/prompt.py, CC BY-SA 4.0) and execution accuracy follows
# evaluation/evaluation_ex.py: a prediction is correct when ``set(pred_rows) == set(gold_rows)``.
#
# Differences from the reference evaluator, all of which only make scoring stricter or safer:
# - the database is opened read-only and an SQLite authorizer only allows reads, so a
#   prediction cannot modify the database used by later samples;
# - a single statement is executed per query;
# - a timed-out query is interrupted inside SQLite instead of being abandoned in a worker.

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_DENIED_FUNCTIONS = {'load_extension', 'readfile', 'writefile', 'edit', 'fts3_tokenizer'}
_FENCE_RE = re.compile(r'```[ \t]*(?:sql|sqlite)?[ \t]*\n?(.*?)```', re.IGNORECASE | re.DOTALL)


@dataclass
class ExecResult:
    rows: Optional[List[Tuple]] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def db_file(db_root: str, db_id: str) -> Path:
    """Path of a BIRD database: ``<db_root>/<db_id>/<db_id>.sqlite``."""
    return Path(db_root) / db_id / f'{db_id}.sqlite'


def extract_sql(text: str) -> str:
    """Return the last fenced code block of a completion, or the whole completion if none."""
    if not text:
        return ''
    blocks = _FENCE_RE.findall(text)
    sql = blocks[-1] if blocks else text
    return sql.strip().rstrip(';').strip()


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and arg2 and arg2.lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _connect_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
    return conn


def execute_sql(path: Path, sql: str, timeout: float = 30.0, max_rows: int = 100_000) -> ExecResult:
    """Run one read-only statement; return its rows or an error message."""
    if not sql:
        return ExecResult(error='empty query')
    if not Path(path).is_file():
        return ExecResult(error=f'database not found: {path}')
    deadline = time.monotonic() + timeout
    conn = None
    try:
        conn = _connect_read_only(Path(path))
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        rows = conn.execute(sql).fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return ExecResult(error=f'result exceeds {max_rows} rows')
        return ExecResult(rows=[tuple(r) for r in rows])
    except sqlite3.OperationalError as e:
        return ExecResult(error='timeout' if 'interrupted' in str(e) else f'OperationalError: {e}')
    except Exception as e:  # sqlite3.Warning for several statements, DatabaseError, ...
        return ExecResult(error=f'{type(e).__name__}: {e}')
    finally:
        if conn is not None:
            conn.close()


def schema_prompt(path: Path, num_rows: int = 0) -> str:
    """CREATE TABLE statements of every table, optionally with ``num_rows`` example rows each."""
    conn = _connect_read_only(Path(path))
    try:
        tables = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
        parts = []
        for name, create_sql in tables:
            if name == 'sqlite_sequence' or not create_sql:
                continue
            if num_rows:
                cursor = conn.execute(f'SELECT * FROM `{name}` LIMIT {int(num_rows)}')
                header = ' '.join(d[0] for d in cursor.description)
                rows = '\n'.join(' '.join(str(v) for v in row) for row in cursor.fetchall())
                create_sql = (
                    f'{create_sql} \n /* \n {num_rows} example rows: \n SELECT * FROM `{name}` LIMIT {num_rows}; \n '
                    f'{header}\n{rows} \n */'
                )
            parts.append(create_sql)
        return '\n\n'.join(parts)
    finally:
        conn.close()


def build_prompt(schema: str, question: str, evidence: Optional[str]) -> str:
    """BIRD reference prompt (schema, question, external knowledge, step-by-step cue) with a fenced answer."""
    knowledge_text = ' and understanding External Knowledge' if evidence else ''
    lines = [
        schema,
        '',
        f'-- Using valid SQLite{knowledge_text}, answer the following questions for the tables provided above.',
        f'-- {question}',
    ]
    if evidence:
        lines.append(f'-- External Knowledge: {evidence}')
    lines += [
        '',
        'Generate the SQLite for the above question after thinking step by step.',
        'Write the final query in a single ```sql code block at the end of your answer.',
    ]
    return '\n'.join(lines)
