from types import SimpleNamespace

from evalscope.backend.rag_eval.cmteb import task_template


class FakeModelMeta:

    revision = 'test-revision'

    @staticmethod
    def model_name_as_path():
        return 'eval__test-model'


class FakeModel:
    mteb_model_meta = FakeModelMeta()


class FakeMainResult:

    task_name = 'CustomRetrieval'
    scores = {
        'test': [
            {
                'hf_subset': 'default',
                'main_score': 0.75,
            }
        ]
    }

    @property
    def task_type(self):
        raise KeyError('CustomRetrieval')


class FakeModelResult:

    @staticmethod
    def only_main_score():
        return FakeMainResult()


def test_build_result_table_uses_evaluated_task_metadata_for_custom_tasks():
    tasks = [
        SimpleNamespace(
            metadata=SimpleNamespace(
                name='CustomRetrieval',
                type='Retrieval',
            )
        )
    ]

    data = task_template._build_result_table(
        FakeModel(),
        [FakeModelResult()],
        task_template._build_task_type_by_name(tasks),
    )

    assert data == [
        {
            'Model': 'test-model',
            'Revision': 'test-revision',
            'Task Type': 'Retrieval',
            'Task': 'CustomRetrieval',
            'Split': 'test',
            'Subset': 'default',
            'Main Score': 0.75,
        }
    ]


def test_resolve_task_type_falls_back_when_mteb_registry_lookup_fails():
    assert task_template._resolve_task_type(FakeMainResult()) == 'Unknown'


def test_resolve_task_type_handles_missing_result_shape():
    assert task_template._resolve_task_type(None) == 'Unknown'
    assert task_template._resolve_task_type(SimpleNamespace()) == 'Unknown'
