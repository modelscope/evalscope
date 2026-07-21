import sys

from evalscope.utils import logger


def test_distributed_env_detection_does_not_import_torch_distributed(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, 'torch.distributed', raising=False)
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('RANK', '1')

    assert logger._is_torch_dist() is True
    assert logger._is_torch_master() is False
    assert 'torch.distributed' not in sys.modules

    monkeypatch.setenv('RANK', '0')

    assert logger._is_torch_master() is True
