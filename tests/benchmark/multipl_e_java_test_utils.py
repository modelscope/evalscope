"""Run fixed, self-contained Java regressions without a sandbox service or external JAR."""

import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest


@lru_cache(maxsize=1)
def java_major_version() -> int:
    """Skip Java integration checks explicitly when a working local JDK is unavailable."""
    if not shutil.which('javac') or not shutil.which('java'):
        pytest.skip('Java integration requires local javac and java; assembly tests still run.')
    result = subprocess.run(['javac', '-version'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        pytest.skip('Java integration requires a working local JDK; assembly tests still run.')
    match = re.search(r'javac (\d+)(?:\.(\d+))?', result.stdout + result.stderr)
    assert match is not None, result.stdout + result.stderr
    return int(match.group(2) if match.group(1) == '1' else match.group(1))


def run_java(code: str) -> dict:
    """Compile Main.java and execute its real assertions with java -ea.

    The arithmetic fixtures never use javatuples. A temporary package marker only
    resolves EvalScope's unconditional wildcard import; it provides no tuple API
    and does not claim to validate the SandboxFusion runtime's external library.
    """
    java_major_version()
    with tempfile.TemporaryDirectory(prefix='evalscope-multipl-java-') as directory:
        workdir = Path(directory)
        source = workdir / 'Main.java'
        source.write_text(code, encoding='utf-8')
        marker = workdir / 'org' / 'javatuples' / 'UnusedImport.java'
        marker.parent.mkdir(parents=True)
        marker.write_text('package org.javatuples; public final class UnusedImport {}\n', encoding='utf-8')
        compiled = subprocess.run(
            ['javac', '-cp', str(workdir), '-d', str(workdir), str(marker), str(source)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if compiled.returncode:
            return {'status': 'error', 'stage': 'compile', 'stdout': compiled.stdout, 'stderr': compiled.stderr}
        result = subprocess.run(
            ['java', '-ea', '-cp', str(workdir), 'Main'], capture_output=True, text=True, timeout=20
        )
        return {
            'status': 'success' if result.returncode == 0 else 'error',
            'stage': 'run',
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
