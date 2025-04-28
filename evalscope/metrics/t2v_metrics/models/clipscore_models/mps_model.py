import torch
from modelscope import AutoProcessor
from transformers import CLIPConfig
from typing import List

from ...constants import CACHE_DIR
from ..model import ScoreModel

MPS_MODELS = ['mps']


class MPSModel(ScoreModel):
    'A wrapper for MPS Score models'

    def __init__(self, model_name='mps', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in MPS_MODELS
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        from ..utils import download_file
        from .build_mps_model.clip_model import CLIPModel

        assert self.model_name == 'mps'

        processor_name_or_path = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)

        config = download_file('AI-ModelScope/MPS', file_name='config.json', cache_dir=self.cache_dir)
        model_pretrained_path = download_file(
            'AI-ModelScope/MPS', file_name='MPS_overall_state_dict.pt', cache_dir=self.cache_dir)  # modelscope model
        model_weight = torch.load(model_pretrained_path, weights_only=True, map_location='cpu')

        self.model = CLIPModel(config=CLIPConfig.from_json_file(config))
        self.model.load_state_dict(model_weight, strict=False)
        self.model.eval().to(self.device)

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (no preprocessing!!) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        image = self.processor(images=image, return_tensors='pt')['pixel_values']
        return image

    def process_text(self, text: List[str]) -> dict:
        """Process the text(s), and return a tensor (after preprocessing) put on self.device
        """
        text_inputs = self.processor(
            text=text,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).input_ids
        return text_inputs

    @torch.no_grad()
    def forward(self, images: List[str], texts: List[str], condition=None) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts)
        image_input = self.load_images(images).to(self.device)
        text_input = self.process_text(texts).to(self.device)
        if condition is None:
            condition = 'light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things.'
        condition_batch = self.process_text(condition).repeat(text_input.shape[0], 1).to(self.device)

        # embed
        text_f, text_features = self.model.model.get_text_features(text_input)

        image_f = self.model.model.get_image_features(image_input.half())
        condition_f, _ = self.model.model.get_text_features(condition_batch)

        sim_text_condition = torch.einsum('b i d, b j d -> b j i', text_f, condition_f)
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
        mask = mask.repeat(1, image_f.shape[1], 1)
        image_features = self.model.cross_model(image_f, text_f, mask.half())[:, 0, :]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_score = self.model.logit_scale.exp() * text_features @ image_features.T

        return image_score[0]
