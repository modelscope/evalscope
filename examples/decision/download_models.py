"""Download a pinned decision release exclusively through ModelScope and verify all files."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Retain the backbone, adapter, head and calibration as a single release."""
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['Cloudflare/clef-flash', 'autotrust/JEV-27B'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', help='Defaults to the validated revision in model_revisions.json.')
    parser.add_argument('--ref', default='master', help='Named SDK download ref; its file hashes must match the commit.')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('model_revisions.json').read_text())[args.model]
    revision = args.revision or manifest['revision']
    if len(revision) != 40:
        raise ValueError('Pin a full ModelScope Git commit.')
    api = HubApi()
    files = api.get_model_files(args.model, revision=revision, recursive=True)
    named_files = api.get_model_files(args.model, revision=args.ref, recursive=True)
    def hashes(entries: list[dict]) -> dict[str, str]:
        return {entry['Path']: entry['Sha256'] for entry in entries if entry['Type'] == 'blob'}
    if hashes(files) != hashes(named_files):
        raise ValueError('The named ModelScope ref no longer matches the pinned commit; use an unchanged tag/ref.')
    # Some ModelScope SDK releases reject commit IDs in snapshot_download's revision validator.
    # Verify the named ref against the immutable tree, then verify every downloaded byte.
    snapshot_download(args.model, revision=args.ref, local_dir=str(args.output))
    checksums = {}
    for entry in files:
        if entry['Type'] != 'blob':
            continue
        path = args.output / entry['Path']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != entry['Sha256']:
            raise ValueError(f'Checksum mismatch: {entry["Path"]}')
        checksums[entry['Path']] = actual
    evidence = {'model': args.model, 'revision': revision, 'files': checksums, 'source': 'modelscope'}
    (args.output / 'evalscope_release_manifest.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'revision': revision, 'verified_files': len(checksums)}))


if __name__ == '__main__':
    main()
