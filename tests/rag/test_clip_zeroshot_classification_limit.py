import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sklearn')
pytest.importorskip('langchain_core')

import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from evalscope.backend.rag_eval.clip_benchmark.tasks import zeroshot_classification  # noqa: E402


class _IdentityModel:
    def encode_image(self, images):
        return images


@pytest.mark.parametrize('limit,expected', [(None, 10), (3, 3), (4, 4), (6, 6), (8, 8), (10, 10), (50, 10)])
def test_run_classification_scores_exactly_limit_samples(limit, expected):
    labels = torch.arange(10) % 3
    loader = DataLoader(TensorDataset(F.one_hot(labels, 3).float(), labels), batch_size=4)

    logits, target = zeroshot_classification.run_classification(
        _IdentityModel(), torch.eye(3), loader, 'cpu', amp=False, limit=limit
    )

    assert len(logits) == len(target) == expected
    assert torch.equal(target, labels[:expected])
