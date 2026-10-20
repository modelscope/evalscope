import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sklearn')
pytest.importorskip('langchain_core')

import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from evalscope.backend.rag_eval.clip_benchmark.tasks import zeroshot_classification  # noqa: E402

CLASSES = ['0', '1', '2', '3', '4']


class _OneHotModel:
    """Embeds `class {i}` and an image labelled i to the same one-hot vector."""

    def encode_text(self, texts):
        return torch.stack([F.one_hot(torch.tensor(int(text.split()[-1])), len(CLASSES)).float() for text in texts])

    def encode_image(self, images):
        return images


class _Dataset(TensorDataset):
    classes = CLASSES


def test_accuracy_returns_python_floats():
    logits = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
    target = torch.tensor([0, 2, 1])

    assert zeroshot_classification.accuracy(logits, target, topk=(1, 2)) == pytest.approx([1 / 3, 2 / 3])


def test_evaluate_scores_every_sample():
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    images = F.one_hot(labels, len(CLASSES)).float()
    images[-1] = F.one_hot(torch.tensor(0), len(CLASSES)).float()  # one wrong prediction out of ten
    loader = DataLoader(_Dataset(images, labels), batch_size=4)

    metrics = zeroshot_classification.evaluate(_OneHotModel(), loader, CLASSES, ['class {c}'], 'cpu', amp=False)

    assert metrics['acc1'] == pytest.approx(0.9)
    assert metrics['acc5'] == pytest.approx(1.0)
