import argparse
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Set
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PARTS = {'__pycache__', '.DS_Store', '.pytest_cache', 'coverage', 'node_modules'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify EvalScope wheel and source distribution contents.')
    parser.add_argument('--dist-dir', type=Path, default=REPO_ROOT / 'dist')
    return parser.parse_args()


def expected_web_files() -> Dict[str, Path]:
    web_root = REPO_ROOT / 'evalscope' / 'web'
    expected = {'evalscope/web/__init__.py': web_root / '__init__.py'}
    expected.update({
        f'evalscope/web/{path.relative_to(web_root).as_posix()}': path
        for path in (web_root / 'dist').rglob('*')
        if path.is_file()
    })
    return expected


def forbidden_files(names: Iterable[str]) -> Set[str]:
    return {name for name in names if FORBIDDEN_PARTS.intersection(Path(name).parts)}


def validate_file_set(actual: Set[str], expected: Set[str], archive_name: str) -> None:
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected or missing:
        raise ValueError(f'{archive_name}: unexpected web files={unexpected}, missing web files={missing}')


def validate_wheel(wheel: Path, expected: Dict[str, Path]) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        actual_web = {name for name in names if name.startswith('evalscope/web/')}
        validate_file_set(actual_web, set(expected), wheel.name)
        mismatched = sorted(name for name, source in expected.items() if archive.read(name) != source.read_bytes())

    forbidden = sorted(forbidden_files(names))
    if forbidden or mismatched:
        raise ValueError(f'{wheel.name}: forbidden files={forbidden}, content mismatches={mismatched}')


def validate_sdist(sdist: Path, expected: Dict[str, Path]) -> None:
    with tarfile.open(sdist, 'r:gz') as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        roots = {name.split('/', 1)[0] for name in members}
        if len(roots) != 1:
            raise ValueError(f'{sdist.name}: expected one archive root, found {sorted(roots)}')
        root = roots.pop()
        prefix = f'{root}/'
        names = {name.removeprefix(prefix) for name in members}
        actual_web = {name for name in names if name.startswith('evalscope/web/')}
        validate_file_set(actual_web, set(expected), sdist.name)
        mismatched = []
        for name, source in expected.items():
            extracted = archive.extractfile(members[f'{prefix}{name}'])
            if extracted is None or extracted.read() != source.read_bytes():
                mismatched.append(name)

    forbidden = sorted(forbidden_files(names))
    if forbidden or mismatched:
        raise ValueError(f'{sdist.name}: forbidden files={forbidden}, content mismatches={sorted(mismatched)}')


def release_archives(dist_dir: Path) -> tuple[Path, Path]:
    wheels = sorted(dist_dir.glob('*.whl'))
    sdists = sorted(dist_dir.glob('*.tar.gz'))
    other = sorted(path.name for path in dist_dir.iterdir() if path not in {*wheels, *sdists})
    if len(wheels) != 1 or len(sdists) != 1 or other:
        raise ValueError(
            f'{dist_dir}: expected exactly one wheel and one sdist; '
            f'wheels={len(wheels)}, sdists={len(sdists)}, other={other}'
        )
    return wheels[0], sdists[0]


def main() -> None:
    args = parse_args()
    expected = expected_web_files()
    wheel, sdist = release_archives(args.dist_dir.resolve())
    validate_wheel(wheel, expected)
    validate_sdist(sdist, expected)
    print(f'Package validation passed: {len(expected)} web files match in {wheel.name} and {sdist.name}.')


if __name__ == '__main__':
    main()
