import torch
from typing import List

from ...constants import CACHE_DIR
from ..model import ScoreModel

IMAGE_REWARD_MODELS = {
    'image-reward-v1': {
        'variant': 'ImageReward-v1.0'
    },
}


class ImageRewardScoreModel(ScoreModel):
    'A wrapper for ImageReward ITMScore (finetuned on human preference) models'

    def __init__(self, model_name='image-reward-v1', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in IMAGE_REWARD_MODELS, f'Model name must be one of {IMAGE_REWARD_MODELS.keys()}'

        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        from ..utils import download_file
        from .image_reward.ImageReward import ImageReward

        self.variant = IMAGE_REWARD_MODELS[self.model_name]['variant']

        self.model_path = download_file('ZhipuAI/ImageReward', file_name='ImageReward.pt', cache_dir=self.cache_dir)
        self.med_config = download_file('ZhipuAI/ImageReward', file_name='med_config.json', cache_dir=self.cache_dir)

        state_dict = torch.load(self.model_path, map_location='cpu')
        self.model = ImageReward(device=self.device, med_config=self.med_config).to(self.device)
        msg = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        image = [self.model.preprocess(image) for image in image]
        assert all(x.shape == image[0].shape for x in image)
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    def forward(self, images: List[str], texts: List[str]) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), 'Number of images and texts must match'
        rewards = torch.zeros(len(texts), dtype=torch.float32).to(self.device)
        images = self.load_images(images)
        for index in range(len(texts)):
            text_input = self.model.blip.tokenizer(
                texts[index], padding='max_length', truncation=True, max_length=35, return_tensors='pt').to(self.device)
            image_embeds = self.model.blip.visual_encoder(images[index].unsqueeze(0))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            text_output = self.model.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
            reward_score = self.model.mlp(txt_features)
            reward_score = (reward_score - self.model.mean) / self.model.std

            rewards[index] = reward_score

        return rewards
