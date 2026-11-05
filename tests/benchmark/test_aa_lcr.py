"""AA-LCR input documents must follow each question's ordered manifest."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.config import TaskConfig


@pytest.fixture
def record() -> dict[str, Any]:
    return {
        'question_id': '1',
        'document_category': 'Reports',
        'document_set_id': 'report_set',
        'data_source_filenames': 'b.txt;a.txt',
        'data_source_urls': 'https://example.com/b;https://example.com/a',
        'question': 'What is in the first document?',
        'answer': 'Beta',
        'input_tokens': 100,
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / 'texts' / 'Reports' / 'report_set'
    folder.mkdir(parents=True)
    (folder / 'a.txt').write_text('  Alpha\n\n中文  \n', encoding='utf-8')
    (folder / 'b.txt').write_text('\nBeta\n', encoding='utf-8')
    return folder


def _adapter(corpus: Path, local_path: Path | None = None, repeats: int = 1) -> AALCRAdapter:
    args: dict[str, Any] = {'extra_params': {'text_dir': str(corpus.parent.parent)}}
    if local_path is not None:
        args['local_path'] = str(local_path)
    adapter = get_benchmark(
        'aa_lcr',
        TaskConfig(model='mock', datasets=['aa_lcr'], repeats=repeats, dataset_args={'aa_lcr': args}),
    )
    # The real load() performs this normalization when using the loader below.
    if local_path is None:
        adapter.text_dir = corpus.parent.parent
    return adapter


@pytest.mark.parametrize(
    ('manifest', 'documents'),
    [
        ('b.txt;a.txt', 'BEGIN DOCUMENT 1:\nBeta\nEND DOCUMENT 1\n\nBEGIN DOCUMENT 2:\nAlpha\n\n中文\nEND DOCUMENT 2'),
        ('a.txt;b.txt', 'BEGIN DOCUMENT 1:\nAlpha\n\n中文\nEND DOCUMENT 1\n\nBEGIN DOCUMENT 2:\nBeta\nEND DOCUMENT 2'),
    ],
)
def test_prompt_follows_each_manifest(
    corpus: Path, record: dict[str, Any], manifest: str, documents: str
) -> None:
    # Opposite orders over the same files expose the bug on any filesystem.
    record['data_source_filenames'] = manifest
    sample = _adapter(corpus).record_to_sample(record)

    assert sample.input[0].content == (
        f'\nBEGIN INPUT DOCUMENTS\n\n{documents}\n\nEND INPUT DOCUMENTS\n\n'
        'Answer the following question using the input documents provided above.\n\n'
        'START QUESTION\n\nWhat is in the first document?\n\nEND QUESTION\n'
    )
    assert sample.target == 'Beta'
    assert sample.metadata == {
        'question': record['question'],
        'data_source_urls': record['data_source_urls'],
        'input_tokens': 100,
    }


def test_unlisted_files_do_not_change_prompt(corpus: Path, record: dict[str, Any]) -> None:
    adapter = _adapter(corpus)
    expected = adapter.record_to_sample(record).input[0].content
    (corpus / 'local_notes.txt').write_text('This is not a source document.', encoding='utf-8')
    (corpus / 'unrelated.bin').write_bytes(b'\xff')
    (corpus / 'subdirectory').mkdir()

    assert adapter.record_to_sample(record).input[0].content == expected


@pytest.mark.parametrize('manifest', [None, '', '  ', 123, 'a.txt;', ';b.txt', 'a.txt;;b.txt'])
def test_invalid_manifest_is_rejected(corpus: Path, record: dict[str, Any], manifest: Any) -> None:
    record['data_source_filenames'] = manifest

    with pytest.raises(ValueError, match='data_source_filenames'):
        _adapter(corpus).record_to_sample(record)


def test_missing_manifest_is_rejected(corpus: Path, record: dict[str, Any]) -> None:
    del record['data_source_filenames']

    with pytest.raises(ValueError, match='data_source_filenames'):
        _adapter(corpus).record_to_sample(record)


@pytest.mark.parametrize('problem', ['missing_file', 'missing_folder', 'directory', 'invalid_utf8'])
def test_required_document_read_failure_is_rejected(
    corpus: Path, record: dict[str, Any], problem: str
) -> None:
    document = corpus / 'b.txt'
    if problem == 'invalid_utf8':
        document.write_bytes(b'\xff')
    else:
        document.unlink()
        if problem == 'missing_folder':
            (corpus / 'a.txt').unlink()
            corpus.rmdir()
        elif problem == 'directory':
            document.mkdir()

    with pytest.raises(ValueError, match='AA-LCR') as exc_info:
        _adapter(corpus).record_to_sample(record)

    assert str(document) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, (OSError, UnicodeDecodeError))


def test_unreadable_document_is_rejected(
    corpus: Path, record: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read_text = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == corpus / 'b.txt':
            raise PermissionError('Permission denied')
        return original_read_text(path, *args, **kwargs)

    # chmod-based permission failures are not portable when tests run as root.
    monkeypatch.setattr(Path, 'read_text', read_text)
    with pytest.raises(ValueError, match='b.txt') as exc_info:
        _adapter(corpus).record_to_sample(record)

    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_empty_document_keeps_its_manifest_position(corpus: Path, record: dict[str, Any]) -> None:
    (corpus / 'b.txt').write_text(' \n', encoding='utf-8')

    prompt = _adapter(corpus).record_to_sample(record).input[0].content

    assert 'BEGIN DOCUMENT 1:\n\nEND DOCUMENT 1\n\nBEGIN DOCUMENT 2:\nAlpha\n\n中文\nEND DOCUMENT 2' in prompt


@pytest.mark.parametrize(
    ('manifest_name', 'extracted_name'),
    [
        ('EU’s.txt', 'EUΓÇÖs.txt'),
        ('you—and.txt', 'youΓÇöand.txt'),
        ('needed – Guide.txt', 'needed ΓÇô Guide.txt'),
        ('Başev.txt', 'Bas╠ºev.txt'),
        ('Başev.txt', 'Bas\u0327ev.txt'),
    ],
)
def test_official_archive_filename_encodings(
    corpus: Path, record: dict[str, Any], manifest_name: str, extracted_name: str
) -> None:
    # The official ZIP stores UTF-8 names without the UTF-8 flag, including an NFD name.
    # zipfile.extractall() therefore produces these CP437-decoded names.
    (corpus / extracted_name).write_text('Required source', encoding='utf-8')
    record['data_source_filenames'] = f'{manifest_name};b.txt'

    prompt = _adapter(corpus).record_to_sample(record).input[0].content

    assert 'BEGIN DOCUMENT 1:\nRequired source\nEND DOCUMENT 1\n\nBEGIN DOCUMENT 2:\nBeta\nEND DOCUMENT 2' in prompt
    assert 'Alpha' not in prompt


@pytest.mark.parametrize('content', [b'Exact name wins', b'\xff'])
def test_exact_filename_takes_precedence_over_legacy_alias(
    corpus: Path, record: dict[str, Any], content: bytes
) -> None:
    (corpus / 'EU’s.txt').write_bytes(content)
    (corpus / 'EUΓÇÖs.txt').write_text('Legacy copy', encoding='utf-8')
    record['data_source_filenames'] = 'EU’s.txt'

    if content == b'\xff':
        with pytest.raises(ValueError) as exc_info:
            _adapter(corpus).record_to_sample(record)
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)
    else:
        prompt = _adapter(corpus).record_to_sample(record).input[0].content
        assert 'Exact name wins' in prompt
        assert 'Legacy copy' not in prompt


@pytest.mark.parametrize('extension', ['csv', 'jsonl'])
def test_local_loader_preserves_manifests_across_repeats(
    tmp_path: Path, corpus: Path, record: dict[str, Any], extension: str
) -> None:
    records = [record, {**record, 'question_id': '2', 'data_source_filenames': 'a.txt;b.txt', 'answer': 'Alpha'}]
    local_path = tmp_path / 'questions'
    local_path.mkdir()
    questions = local_path / f'test.{extension}'
    if extension == 'csv':
        with questions.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(records)
    else:
        questions.write_text('\n'.join(json.dumps(row) for row in records), encoding='utf-8')
    adapter = _adapter(corpus, local_path, repeats=2)

    dataset = adapter.load_dataset()['default']

    assert len(dataset) == 4
    assert [sample.target for sample in dataset] == ['Beta', 'Beta', 'Alpha', 'Alpha']
    assert [sample.input[0].content.split('BEGIN DOCUMENT 1:\n')[1].splitlines()[0] for sample in dataset] == [
        'Beta', 'Beta', 'Alpha', 'Alpha'
    ]
    assert adapter.benchmark_meta.evaluation_version == 'v1.1'


def test_local_loader_aborts_on_incomplete_corpus(tmp_path: Path, corpus: Path, record: dict[str, Any]) -> None:
    local_path = tmp_path / 'questions'
    local_path.mkdir()
    questions = local_path / 'test.jsonl'
    records = [{**record, 'data_source_filenames': 'a.txt'}, record]
    questions.write_text('\n'.join(json.dumps(row) for row in records), encoding='utf-8')
    (corpus / 'b.txt').unlink()

    with pytest.raises(ValueError, match='b.txt'):
        _adapter(corpus, local_path).load_dataset()
