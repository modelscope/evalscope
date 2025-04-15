import torch
from typing import List

from ...constants import CACHE_DIR
from ..model import ScoreModel

HPSV2_MODELS = ['hpsv2']


class HPSV2ScoreModel(ScoreModel):
    'A wrapper for HPSv2 models '

    def __init__(self, model_name='openai:ViT-L-14', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in HPSV2_MODELS
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        import hpsv2
        self.hpsv2 = hpsv2

    def load_images(self, image: List[str]):
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        images = [self.image_loader(x) for x in image]
        return images

    @torch.no_grad()
    def forward(self, images: List[str], texts: List[str]) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts)
        images = self.load_images(images)
        scores = torch.zeros(len(images), dtype=torch.float16).to(self.device)
        for i in range(len(images)):
            caption = texts[i]
            image = images[i]
            scores[i] = float(self.hpsv2.score(image, caption)[0])

        # return cosine similarity as scores
        return scores
