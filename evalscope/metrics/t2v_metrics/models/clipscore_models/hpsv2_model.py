import torch
from typing import List

from ...constants import CACHE_DIR
from ..model import ScoreModel

HPSV2_MODELS = ['hpsv2', 'hpsv2.1']
HPS_VERSION_MAP = {
    'hpsv2': 'HPS_v2_compressed.pt',
    'hpsv2.1': 'HPS_v2.1_compressed.pt',
}


class HPSV2ScoreModel(ScoreModel):
    'A wrapper for HPSv2 models '

    def __init__(self, model_name='openai:ViT-L-14', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in HPSV2_MODELS
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        import open_clip

        from ..utils import download_file, download_open_clip_model

        self.pretrained, self.arch = 'laion2B-s32B-b79K:ViT-H-14'.split(':')
        # load model from modelscope
        model_file_path = download_open_clip_model(self.arch, self.pretrained, self.cache_dir)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.arch,
            pretrained=model_file_path,
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            image_resize_mode='longest',
            aug_cfg={},
            output_dict=True)

        # update weight
        model_weight_path = download_file('AI-ModelScope/HPSv2', HPS_VERSION_MAP[self.model_name], self.cache_dir)
        checkpoint = torch.load(model_weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = open_clip.get_tokenizer(self.arch)
        self.model.eval()

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
            # Process the image
            image = self.preprocess(image).unsqueeze(0).to(device=self.device, non_blocking=True)
            # Process the prompt
            text = self.tokenizer([caption]).to(device=self.device, non_blocking=True)  # Updated to use texts[i]
            # Calculate the HPS
            with torch.amp.autocast(device_type=self.device):
                outputs = self.model(image, text)
                image_features, text_features = outputs['image_features'], outputs['text_features']
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            scores[i] = float(hps_score[0])

        # return cosine similarity as scores
        return scores
