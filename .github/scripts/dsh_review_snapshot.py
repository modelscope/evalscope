"""Build the read-only DSH workspace from Git objects without checking out PR code."""

import os
import re
import shutil
import subprocess
from pathlib import Path

EVIDENCE = '.evalscope-review'
MAX_TREE_BYTES = 256 * 1024 * 1024


def materialize(sha: str, root: Path) -> None:
    """Copy exact regular Git blobs without running checkout filters or following links."""
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise ValueError('A full PR commit SHA is required.')
    listing = subprocess.check_output(['git', 'ls-tree', '-rlz', sha])
    entries = []
    total = 0
    for record in listing.rstrip(b'\0').split(b'\0'):
        metadata, path = record.split(b'\t', 1)
        mode, kind, blob, size = metadata.split()
        parts = os.fsdecode(path).split('/')
        if any(part.lower() in ('', '.', '..', '.git', EVIDENCE) for part in parts):
            raise ValueError('The review tree contains a reserved or unsafe path.')
        if mode not in (b'100644', b'100755') or kind != b'blob':
            # Symlinks and submodules are not materialized from untrusted PRs.
            continue
        total += int(size)
        entries.append((root.joinpath(*parts), blob, int(size)))
    if total > MAX_TREE_BYTES:
        raise ValueError('The PR tree exceeds the review workspace size limit.')

    root.mkdir()
    with subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        assert process.stdin is not None and process.stdout is not None
        try:
            for path, blob, size in entries:
                process.stdin.write(blob + b'\n')
                process.stdin.flush()
                if process.stdout.readline().split() != [blob, b'blob', str(size).encode()]:
                    raise ValueError('Git returned a different review blob.')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as target:
                    remaining = size
                    while remaining:
                        chunk = process.stdout.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError('Git returned an incomplete review blob.')
                        target.write(chunk)
                        remaining -= len(chunk)
                if process.stdout.read(1) != b'\n':
                    raise ValueError('Git returned an incomplete review blob.')
            process.stdin.close()
            if process.wait():
                raise RuntimeError('Git failed while preparing the review workspace.')
        finally:
            if process.poll() is None:
                process.kill()


def main() -> None:
    """Fetch the PR ref, verify it still matches the selected SHA, and stage evidence."""
    number = int(os.environ['PR_NUMBER'])
    sha = os.environ['PR_SHA']
    directory = Path(os.environ['RUNNER_TEMP']) / 'evalscope-review'
    subprocess.run(
        [
            'git',
            '-c',
            'credential.helper=',
            '-c',
            'credential.helper=!gh auth git-credential',
            'fetch',
            '--no-tags',
            '--no-recurse-submodules',
            '--depth=1',
            'origin',
            f'refs/pull/{number}/head',
        ],
        check=True,
    )
    if subprocess.check_output(['git', 'rev-parse', 'FETCH_HEAD'], text=True).strip() != sha:
        raise RuntimeError('The pull request head changed before its review workspace was prepared.')
    workspace = directory / 'workspace'
    materialize(sha, workspace)
    evidence = workspace / EVIDENCE
    evidence.mkdir()
    for name in ('changes.diff', 'ci.json'):
        shutil.copyfile(directory / name, evidence / name)


if __name__ == '__main__':
    main()
