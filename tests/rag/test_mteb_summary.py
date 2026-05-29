import importlib.util
import sys
import types
from pathlib import Path


def _load_task_template():
    module_path = (
        Path(__file__).parents[2]
        / 'evalscope'
        / 'backend'
        / 'rag_eval'
        / 'cmteb'
        / 'task_template.py'
    )
    stubs = {
        'mteb': types.SimpleNamespace(MTEB=object),
        'tabulate': types.SimpleNamespace(tabulate=lambda *args, **kwargs: ''),
        'evalscope': types.ModuleType('evalscope'),
        'evalscope.backend': types.ModuleType('evalscope.backend'),
        'evalscope.backend.rag_eval': types.SimpleNamespace(
            EmbeddingModel=object, cmteb=object
        ),
        'evalscope.utils': types.ModuleType('evalscope.utils'),
        'evalscope.utils.logger': types.SimpleNamespace(
            get_logger=lambda: types.SimpleNamespace(info=lambda *args, **kwargs: None)
        ),
    }
    old_modules = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            'task_template_under_test', module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class _CustomMainResult:
    task_name = 'CustomRetrieval'

    @property
    def task_type(self):
        raise KeyError("'CustomRetrieval' not found")


def test_custom_task_type_uses_runtime_metadata():
    task_template = _load_task_template()
    task = types.SimpleNamespace(
        metadata=types.SimpleNamespace(name='CustomRetrieval', type='Retrieval')
    )
    task_types = {task.metadata.name: task.metadata.type}

    assert task_template._get_task_type(_CustomMainResult(), task_types) == 'Retrieval'
