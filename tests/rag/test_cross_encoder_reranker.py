import json
import os
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sentence_transformers')

from evalscope.backend.rag_eval.models.reranker import _check_logit_score_support


def _write_modules_json(tmp_path, module_types):
    modules = [{
        'idx': idx,
        'name': str(idx),
        'path': '' if 'Transformer' in module_type else f'{idx}_{module_type}',
        'type': f'sentence_transformers.cross_encoder.modules.{module_type.lower()}.{module_type}',
    } for idx, module_type in enumerate(module_types)]
    with open(os.path.join(tmp_path, 'modules.json'), 'w') as f:
        json.dump(modules, f)
    return str(tmp_path)


def test_no_modules_json_is_a_no_op(tmp_path):
    # A plain BERT-style CrossEncoder checkpoint has no modules.json at all.
    _check_logit_score_support(str(tmp_path))


def test_non_logit_score_modules_json_is_a_no_op(tmp_path, monkeypatch):
    model_dir = _write_modules_json(tmp_path, ['Transformer', 'Dense'])
    monkeypatch.setattr('sentence_transformers.__version__', '3.0.0')
    _check_logit_score_support(model_dir)


def test_logit_score_requires_min_sentence_transformers_version(tmp_path, monkeypatch):
    model_dir = _write_modules_json(tmp_path, ['Transformer', 'LogitScore'])
    monkeypatch.setattr('sentence_transformers.__version__', '5.3.0')
    with pytest.raises(ImportError, match='LogitScore'):
        _check_logit_score_support(model_dir)


def test_logit_score_supported_on_recent_sentence_transformers(tmp_path, monkeypatch):
    model_dir = _write_modules_json(tmp_path, ['Transformer', 'LogitScore'])
    monkeypatch.setattr('sentence_transformers.__version__', '5.4.0')
    _check_logit_score_support(model_dir)


def test_logit_score_rejects_unparseable_version(tmp_path, monkeypatch):
    model_dir = _write_modules_json(tmp_path, ['Transformer', 'LogitScore'])
    monkeypatch.setattr('sentence_transformers.__version__', 'not-a-version')
    with pytest.raises(ImportError, match='LogitScore'):
        _check_logit_score_support(model_dir)
