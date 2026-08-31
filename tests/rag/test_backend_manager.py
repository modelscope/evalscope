import pytest

from evalscope.backend.rag_eval.backend_manager import _SENTENCE_TRANSFORMERS_REASON, require_min_version

ST_ARGS = ('sentence-transformers', '5.4.0', 'sentence-transformers>=5.4.0')


def _require_st(installed_version):
    package, min_version, install_spec = ST_ARGS
    require_min_version(package, installed_version, min_version, install_spec, reason=_SENTENCE_TRANSFORMERS_REASON)


def test_below_floor_rejected():
    with pytest.raises(ImportError, match='LogitScore'):
        _require_st('5.3.0')


def test_floor_accepted():
    _require_st('5.4.0')


def test_prerelease_of_floor_accepted():
    # A dev/rc build of 5.4.0 already ships LogitScore, so the floor is the earliest 5.4.0
    # pre-release; comparing against Version('5.4.0') would reject these.
    _require_st('5.4.0.dev0')
    _require_st('5.4.0rc1')


def test_two_component_version_accepted():
    # '5.4' normalizes to 5.4.0, so it must not be treated as older.
    _require_st('5.4')


def test_newer_version_accepted():
    _require_st('5.6.0')


def test_unparseable_version_rejected():
    with pytest.raises(ImportError, match='LogitScore'):
        _require_st('not-a-version')


def test_error_message_names_package_and_install_spec():
    with pytest.raises(ImportError) as excinfo:
        _require_st('5.3.0')
    message = str(excinfo.value)
    assert 'sentence-transformers >= 5.4.0 is required (got 5.3.0)' in message
    assert 'pip install "sentence-transformers>=5.4.0"' in message


def test_reason_omitted_when_not_given():
    # The mteb/ragas gates pass no reason, so the message must not leave a double space.
    with pytest.raises(ImportError) as excinfo:
        require_min_version('MTEB', '2.6.0', '2.7.0', 'mteb>=2.7.0,<3.0.0')
    message = str(excinfo.value)
    assert message == (
        'MTEB >= 2.7.0 is required (got 2.6.0). Please upgrade: pip install "mteb>=2.7.0,<3.0.0"'
    )


@pytest.mark.parametrize('installed_version', ['2.7.0', '2.7.0rc1', '2.7', '2.8.0', '3.0.0'])
def test_mteb_floor_accepts_supported_versions(installed_version):
    require_min_version('MTEB', installed_version, '2.7.0', 'mteb>=2.7.0,<3.0.0')


@pytest.mark.parametrize('installed_version', ['0.4.0', '0.4.0.dev1', '0.4.1', '0.5.0'])
def test_ragas_floor_accepts_supported_versions(installed_version):
    require_min_version('RAGAS', installed_version, '0.4.0', 'ragas>=0.4.0,<0.5.0')


@pytest.mark.parametrize('installed_version', ['0.3.9', '', 'not-a-version'])
def test_ragas_floor_rejects_unsupported_versions(installed_version):
    with pytest.raises(ImportError, match='RAGAS >= 0.4.0'):
        require_min_version('RAGAS', installed_version, '0.4.0', 'ragas>=0.4.0,<0.5.0')
