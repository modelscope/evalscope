import torch
from abc import abstractmethod
from modelscope import AutoModel, AutoProcessor
from typing import List

from ...constants import CACHE_DIR
from ..model import ScoreModel

PICKSCORE_MODELS = ['pickscore-v1']


class PickScoreModel(ScoreModel):
    'A wrapper for PickScore models'

    def __init__(self, model_name='pickscore-v1', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in PICKSCORE_MODELS
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        assert self.model_name == 'pickscore-v1'
        processor_name_or_path = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        # model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        model_pretrained_name_or_path = 'AI-ModelScope/PickScore_v1'  # modelscope model

        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(self.device)

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (no preprocessing!!) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        image = self.processor(
            images=image, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device)
        # image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    def forward(self, images: List[str], texts: List[str]) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts)
        image = self.load_images(images)
        text_inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt',
        ).to(self.device)

        # embed
        image_embs = self.model.get_image_features(**image)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = (image_embs * text_embs).sum(dim=-1)
        return scores
