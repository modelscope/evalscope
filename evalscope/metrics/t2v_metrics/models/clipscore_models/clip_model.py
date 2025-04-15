import open_clip
import torch
from abc import abstractmethod
from typing import List

from ...constants import CACHE_DIR
from ..model import ScoreModel

CLIP_MODELS = [f'{pretrained}:{arch}' for arch, pretrained in open_clip.list_pretrained()]


class CLIPScoreModel(ScoreModel):
    "A wrapper for OpenCLIP models (including openAI's CLIP, OpenCLIP, DatacompCLIP)"

    def __init__(self, model_name='openai:ViT-L-14', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in CLIP_MODELS
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        self.pretrained, self.arch = self.model_name.split(':')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.arch, pretrained=self.pretrained, device=self.device, cache_dir=self.cache_dir)
        self.tokenizer = open_clip.get_tokenizer(self.arch)
        self.model.eval()

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        image = [self.preprocess(x) for x in image]
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    def forward(self, images: List[str], texts: List[str]) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts)
        image = self.load_images(images)
        text = self.tokenizer(texts).to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # return cosine similarity as scores
        return (image_features * text_features).sum(dim=-1)
