"""Unit tests for the device-detection helpers used across the local-serving
and perf paths.

These cover the NPU/CUDA/CPU device selection that ``get_device()`` already
exposes, plus the small NPU-aware branches added to ``update_gpu_usage`` and
``seed_everything``. They run on a CPU-only box by monkeypatching the
``transformers.utils.is_torch_*_available`` probes that ``get_device`` reads.
"""

import pytest

from evalscope.utils.model_utils import get_device, seed_everything  # noqa: E402


def _patch_probes(monkeypatch, *, npu=False, mps=False, cuda=False):
    import transformers.utils as tutils

    monkeypatch.setattr(tutils, 'is_torch_npu_available', lambda: npu, raising=False)
    monkeypatch.setattr(tutils, 'is_torch_mps_available', lambda: mps, raising=False)
    monkeypatch.setattr(tutils, 'is_torch_cuda_available', lambda: cuda, raising=False)


def test_get_device_prefers_npu(monkeypatch):
    _patch_probes(monkeypatch, npu=True, mps=True, cuda=True)
    assert get_device() == 'npu'


def test_get_device_prefers_mps_over_cuda(monkeypatch):
    _patch_probes(monkeypatch, npu=False, mps=True, cuda=True)
    assert get_device() == 'mps'


def test_get_device_falls_back_to_cuda(monkeypatch):
    _patch_probes(monkeypatch, npu=False, mps=False, cuda=True)
    assert get_device() == 'cuda'


def test_get_device_falls_back_to_cpu(monkeypatch):
    _patch_probes(monkeypatch, npu=False, mps=False, cuda=False)
    assert get_device() == 'cpu'


def test_update_gpu_usage_handles_cpu_only(monkeypatch):
    """On a CPU-only box update_gpu_usage must stay at 0 and not crash."""
    from evalscope.perf.utils.benchmark_util import BenchmarkData

    _patch_probes(monkeypatch, npu=False, mps=False, cuda=False)
    m = BenchmarkData()
    m.update_gpu_usage()
    assert m.max_gpu_memory_cost == 0


def test_update_gpu_usage_reads_npu_when_available(monkeypatch):
    """When NPU is available, update_gpu_usage should query torch.npu,
    not torch.cuda."""
    import torch

    from evalscope.perf.utils.benchmark_util import BenchmarkData

    _patch_probes(monkeypatch, npu=True, mps=False, cuda=False)

    fake_npu = type('fake_npu', (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.device_count = lambda: 2
    fake_npu.max_memory_allocated = lambda i: float(i + 1) * 1024**3  # 1 GiB, 2 GiB
    monkeypatch.setattr(torch, 'npu', fake_npu, raising=False)

    m = BenchmarkData()
    m.update_gpu_usage()
    # 1 + 2 = 3 GiB
    assert m.max_gpu_memory_cost == pytest.approx(3.0)


def test_seed_everything_sets_npu_seed_when_available(monkeypatch):
    """seed_everything should call torch.npu.manual_seed_all on NPU."""
    import torch

    calls = {'npu_seed': None}

    fake_npu = type('fake_npu', (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.manual_seed_all = lambda seed: calls.__setitem__('npu_seed', seed)
    monkeypatch.setattr(torch, 'npu', fake_npu, raising=False)

    seed_everything(1234)
    assert calls['npu_seed'] == 1234
