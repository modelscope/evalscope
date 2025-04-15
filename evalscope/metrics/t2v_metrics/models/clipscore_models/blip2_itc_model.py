import os
import torch
from torchvision import transforms
from typing import List

from ...constants import CACHE_DIR
from ..model import ScoreModel
from ..vqascore_models.lavis.models import load_model

BLIP2_ITC_MODELS = {
    'blip2-itc': {
        'variant': 'pretrain'
    },
    'blip2-itc-vitL': {
        'variant': 'pretrain_vitL'
    },
    'blip2-itc-coco': {
        'variant': 'coco'
    },
}


class BLIP2ITCScoreModel(ScoreModel):
    'A wrapper for BLIP-2 ITCScore models'

    def __init__(self, model_name='blip2-itc', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in BLIP2_ITC_MODELS, f'Model name must be one of {BLIP2_ITC_MODELS.keys()}'
        os.environ['TORCH_HOME'] = cache_dir
        import timm.models.hub as timm_hub
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        self.variant = BLIP2_ITC_MODELS[self.model_name]['variant']
        self.model = load_model('blip2', self.variant, is_eval=True, device=self.device)
        if self.variant == 'coco':
            size = 364
        else:
            size = 224
        self.image_preprocess = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        image = [self.image_preprocess(image) for image in image]
        assert all(x.shape == image[0].shape for x in image)
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, images: List[str], texts: List[str]) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), 'Number of images and texts must match'

        images = self.load_images(images)
        image_feat = self.model.ln_vision(self.model.visual_encoder(images))
        image_att = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(self.device)
        query_token = self.model.query_tokens.expand(image_feat.shape[0], -1, -1)
        query_att = torch.ones(query_token.size()[:-1], dtype=torch.long).to(query_token.device)
        query_output = self.model.Qformer.bert(
            query_embeds=query_token,
            encoder_hidden_states=image_feat,
            encoder_attention_mask=image_att,
            use_cache=True,
            return_dict=True,
        )
        image_embed = self.model.vision_proj(query_output.last_hidden_state)  # B x D
        image_embed = torch.nn.functional.normalize(image_embed, dim=-1)

        text_input = self.model.tokenizer(
            texts, padding='max_length', truncation=True, max_length=35, return_tensors='pt').to(self.device)
        text_output = self.model.Qformer.bert(
            text_input.input_ids, attention_mask=text_input.attention_mask, return_dict=True)
        text_embed = torch.nn.functional.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        itc_scores = (image_embed * text_embed.unsqueeze(1)).sum(dim=-1)
        itc_scores = itc_scores.max(1)[
            0]  # This follows the practice in BLIP2 codebase -- only take the max of the logits across query
        return itc_scores
