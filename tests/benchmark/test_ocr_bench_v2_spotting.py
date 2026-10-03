import builtins
import zipfile

from evalscope.benchmarks.ocr_bench.ocr_bench_v2 import spotting_metric


def test_spotting_files_are_written_as_utf8(tmp_path, monkeypatch):
    """Regression: the spotting scorer decodes the gt/submission files as UTF-8, so they must be written as UTF-8.

    With the locale default encoding (cp1252, cp1254, ... on Windows) a recognised text such as
    'Café → Exit' raised UnicodeEncodeError before scoring started.
    """

    def ansi_open(file, mode='r', *args, **kwargs):
        if 'b' not in mode:
            kwargs.setdefault('encoding', 'cp1252')
        return builtins.open(file, mode, *args, **kwargs)

    received = {}

    def fake_main_evaluation(params, *args, **kwargs):
        for key in ('g', 's'):
            with zipfile.ZipFile(params[key]) as zf:
                received[key] = spotting_metric.rrc_evaluation_funcs.decode_utf8(zf.read(zf.namelist()[0]))
        return {'method': {'hmean': 1.0}}

    monkeypatch.setattr(spotting_metric, 'open', ansi_open, raising=False)
    monkeypatch.setattr(spotting_metric, 'DEFAULT_EVALSCOPE_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(spotting_metric.rrc_evaluation_funcs, 'main_evaluation', fake_main_evaluation)

    text = 'Café → Exit'
    doc = {'bbox_list': [[10, 10, 100, 10, 100, 40, 10, 40]], 'content': [text]}

    assert spotting_metric.spotting_evaluation([[10, 10, 100, 40, text]], doc) == 1.0
    assert received['g'].endswith(text)
    assert received['s'].endswith(text)
