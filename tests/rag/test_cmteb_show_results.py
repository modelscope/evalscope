import pytest

pytest.importorskip('mteb')

from evalscope.backend.rag_eval.cmteb.task_template import show_results


class MockModelMeta:

    revision = 'v1'

    def model_name_as_path(self):
        return 'eval__test-model'


class MockModel:

    mteb_model_meta = MockModelMeta()


class MockMainScore:

    def __init__(self, task_name, raise_key_error=False):
        self.task_name = task_name
        self.scores = {'test': [{'hf_subset': 'default', 'main_score': 0.5}]}
        self._raise_key_error = raise_key_error

    @property
    def task_type(self):
        if self._raise_key_error:
            raise KeyError(f"KeyError: '{self.task_name}' not found. Did you mean: EcomRetrieval?")
        return 'Retrieval'


class MockModelResult:

    def __init__(self, main_score):
        self._main_score = main_score

    def only_main_score(self):
        return self._main_score


def test_show_results_prefers_local_task_type_map(tmp_path):
    # Custom task resolves via the local map, even though mteb lookup would raise KeyError.
    results = [MockModelResult(MockMainScore('CustomRetrieval', raise_key_error=True))]

    data = show_results(str(tmp_path), MockModel(), results, {'CustomRetrieval': 'Retrieval'})

    assert data[0]['Task Type'] == 'Retrieval'
    assert data[0]['Task'] == 'CustomRetrieval'


def test_show_results_falls_back_to_custom_on_keyerror(tmp_path):
    # Regression for issue #915: not in local map and mteb lookup raises KeyError.
    results = [MockModelResult(MockMainScore('CustomRetrieval', raise_key_error=True))]

    data = show_results(str(tmp_path), MockModel(), results, task_type_map=None)

    assert data[0]['Task Type'] == 'Custom'
