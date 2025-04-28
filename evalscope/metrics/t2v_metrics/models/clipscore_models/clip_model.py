import os
import torch
from typing import List

from ...constants import CACHE_DIR
from ..model import ScoreModel

CLIP_MODELS = [
    'openai:RN50', 'yfcc15m:RN50', 'cc12m:RN50', 'openai:RN101', 'yfcc15m:RN101', 'openai:RN50x4', 'openai:RN50x16',
    'openai:RN50x64', 'openai:ViT-B-32', 'laion400m_e31:ViT-B-32', 'laion400m_e32:ViT-B-32', 'laion2b_e16:ViT-B-32',
    'laion2b_s34b_b79k:ViT-B-32', 'datacomp_xl_s13b_b90k:ViT-B-32', 'datacomp_m_s128m_b4k:ViT-B-32',
    'commonpool_m_clip_s128m_b4k:ViT-B-32', 'commonpool_m_laion_s128m_b4k:ViT-B-32',
    'commonpool_m_image_s128m_b4k:ViT-B-32', 'commonpool_m_text_s128m_b4k:ViT-B-32',
    'commonpool_m_basic_s128m_b4k:ViT-B-32', 'commonpool_m_s128m_b4k:ViT-B-32', 'datacomp_s_s13m_b4k:ViT-B-32',
    'commonpool_s_clip_s13m_b4k:ViT-B-32', 'commonpool_s_laion_s13m_b4k:ViT-B-32',
    'commonpool_s_image_s13m_b4k:ViT-B-32', 'commonpool_s_text_s13m_b4k:ViT-B-32',
    'commonpool_s_basic_s13m_b4k:ViT-B-32', 'commonpool_s_s13m_b4k:ViT-B-32', 'metaclip_400m:ViT-B-32',
    'metaclip_fullcc:ViT-B-32', 'datacomp_s34b_b86k:ViT-B-32-256', 'openai:ViT-B-16', 'laion400m_e31:ViT-B-16',
    'laion400m_e32:ViT-B-16', 'laion2b_s34b_b88k:ViT-B-16', 'datacomp_xl_s13b_b90k:ViT-B-16',
    'datacomp_l_s1b_b8k:ViT-B-16', 'commonpool_l_clip_s1b_b8k:ViT-B-16', 'commonpool_l_laion_s1b_b8k:ViT-B-16',
    'commonpool_l_image_s1b_b8k:ViT-B-16', 'commonpool_l_text_s1b_b8k:ViT-B-16', 'commonpool_l_basic_s1b_b8k:ViT-B-16',
    'commonpool_l_s1b_b8k:ViT-B-16', 'dfn2b:ViT-B-16', 'metaclip_400m:ViT-B-16', 'metaclip_fullcc:ViT-B-16',
    'laion400m_e31:ViT-B-16-plus-240', 'laion400m_e32:ViT-B-16-plus-240', 'openai:ViT-L-14', 'laion400m_e31:ViT-L-14',
    'laion400m_e32:ViT-L-14', 'laion2b_s32b_b82k:ViT-L-14', 'datacomp_xl_s13b_b90k:ViT-L-14',
    'commonpool_xl_clip_s13b_b90k:ViT-L-14', 'commonpool_xl_laion_s13b_b90k:ViT-L-14',
    'commonpool_xl_s13b_b90k:ViT-L-14', 'metaclip_400m:ViT-L-14', 'metaclip_fullcc:ViT-L-14', 'dfn2b:ViT-L-14',
    'dfn2b_s39b:ViT-L-14', 'openai:ViT-L-14-336', 'laion2b_s32b_b79k:ViT-H-14', 'metaclip_fullcc:ViT-H-14',
    'metaclip_altogether:ViT-H-14', 'dfn5b:ViT-H-14', 'dfn5b:ViT-H-14-378', 'laion2b_s12b_b42k:ViT-g-14',
    'laion2b_s34b_b88k:ViT-g-14', 'laion2b_s39b_b160k:ViT-bigG-14', 'metaclip_fullcc:ViT-bigG-14',
    'laion2b_s12b_b32k:roberta-ViT-B-32', 'laion5b_s13b_b90k:xlm-roberta-base-ViT-B-32',
    'frozen_laion5b_s13b_b90k:xlm-roberta-large-ViT-H-14', 'laion400m_s13b_b51k:convnext_base',
    'laion2b_s13b_b82k:convnext_base_w', 'laion2b_s13b_b82k_augreg:convnext_base_w',
    'laion_aesthetic_s13b_b82k:convnext_base_w', 'laion_aesthetic_s13b_b82k:convnext_base_w_320',
    'laion_aesthetic_s13b_b82k_augreg:convnext_base_w_320', 'laion2b_s26b_b102k_augreg:convnext_large_d',
    'laion2b_s29b_b131k_ft:convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup:convnext_large_d_320',
    'laion2b_s34b_b82k_augreg:convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind:convnext_xxlarge',
    'laion2b_s34b_b82k_augreg_soup:convnext_xxlarge', 'laion2b_s13b_b90k:coca_ViT-B-32',
    'mscoco_finetuned_laion2b_s13b_b90k:coca_ViT-B-32', 'laion2b_s13b_b90k:coca_ViT-L-14',
    'mscoco_finetuned_laion2b_s13b_b90k:coca_ViT-L-14', 'laion400m_s11b_b41k:EVA01-g-14',
    'merged2b_s11b_b114k:EVA01-g-14-plus', 'merged2b_s8b_b131k:EVA02-B-16', 'merged2b_s4b_b131k:EVA02-L-14',
    'merged2b_s6b_b61k:EVA02-L-14-336', 'laion2b_s4b_b115k:EVA02-E-14', 'laion2b_s9b_b144k:EVA02-E-14-plus',
    'webli:ViT-B-16-SigLIP', 'webli:ViT-B-16-SigLIP-256', 'webli:ViT-B-16-SigLIP-i18n-256', 'webli:ViT-B-16-SigLIP-384',
    'webli:ViT-B-16-SigLIP-512', 'webli:ViT-L-16-SigLIP-256', 'webli:ViT-L-16-SigLIP-384', 'webli:ViT-SO400M-14-SigLIP',
    'webli:ViT-SO400M-16-SigLIP-i18n-256', 'webli:ViT-SO400M-14-SigLIP-378', 'webli:ViT-SO400M-14-SigLIP-384',
    'webli:ViT-B-32-SigLIP2-256', 'webli:ViT-B-16-SigLIP2', 'webli:ViT-B-16-SigLIP2-256', 'webli:ViT-B-16-SigLIP2-384',
    'webli:ViT-B-16-SigLIP2-512', 'webli:ViT-L-16-SigLIP2-256', 'webli:ViT-L-16-SigLIP2-384',
    'webli:ViT-L-16-SigLIP2-512', 'webli:ViT-SO400M-14-SigLIP2', 'webli:ViT-SO400M-14-SigLIP2-378',
    'webli:ViT-SO400M-16-SigLIP2-256', 'webli:ViT-SO400M-16-SigLIP2-384', 'webli:ViT-SO400M-16-SigLIP2-512',
    'webli:ViT-gopt-16-SigLIP2-256', 'webli:ViT-gopt-16-SigLIP2-384', 'datacomp1b:ViT-L-14-CLIPA',
    'datacomp1b:ViT-L-14-CLIPA-336', 'datacomp1b:ViT-H-14-CLIPA', 'laion2b:ViT-H-14-CLIPA-336',
    'datacomp1b:ViT-H-14-CLIPA-336', 'datacomp1b:ViT-bigG-14-CLIPA', 'datacomp1b:ViT-bigG-14-CLIPA-336',
    'v1:nllb-clip-base', 'v1:nllb-clip-large', 'v1:nllb-clip-base-siglip', 'mrl:nllb-clip-base-siglip',
    'v1:nllb-clip-large-siglip', 'mrl:nllb-clip-large-siglip', 'datacompdr:MobileCLIP-S1', 'datacompdr:MobileCLIP-S2',
    'datacompdr:MobileCLIP-B', 'datacompdr_lt:MobileCLIP-B', 'datacomp1b:ViTamin-S', 'datacomp1b:ViTamin-S-LTT',
    'datacomp1b:ViTamin-B', 'datacomp1b:ViTamin-B-LTT', 'datacomp1b:ViTamin-L', 'datacomp1b:ViTamin-L-256',
    'datacomp1b:ViTamin-L-336', 'datacomp1b:ViTamin-L-384', 'datacomp1b:ViTamin-L2', 'datacomp1b:ViTamin-L2-256',
    'datacomp1b:ViTamin-L2-336', 'datacomp1b:ViTamin-L2-384', 'datacomp1b:ViTamin-XL-256', 'datacomp1b:ViTamin-XL-336',
    'datacomp1b:ViTamin-XL-384', 'openai:RN50-quickgelu', 'yfcc15m:RN50-quickgelu', 'cc12m:RN50-quickgelu',
    'openai:RN101-quickgelu', 'yfcc15m:RN101-quickgelu', 'openai:RN50x4-quickgelu', 'openai:RN50x16-quickgelu',
    'openai:RN50x64-quickgelu', 'openai:ViT-B-32-quickgelu', 'laion400m_e31:ViT-B-32-quickgelu',
    'laion400m_e32:ViT-B-32-quickgelu', 'metaclip_400m:ViT-B-32-quickgelu', 'metaclip_fullcc:ViT-B-32-quickgelu',
    'openai:ViT-B-16-quickgelu', 'dfn2b:ViT-B-16-quickgelu', 'metaclip_400m:ViT-B-16-quickgelu',
    'metaclip_fullcc:ViT-B-16-quickgelu', 'openai:ViT-L-14-quickgelu', 'metaclip_400m:ViT-L-14-quickgelu',
    'metaclip_fullcc:ViT-L-14-quickgelu', 'dfn2b:ViT-L-14-quickgelu', 'openai:ViT-L-14-336-quickgelu',
    'metaclip_fullcc:ViT-H-14-quickgelu', 'dfn5b:ViT-H-14-quickgelu', 'dfn5b:ViT-H-14-378-quickgelu',
    'metaclip_fullcc:ViT-bigG-14-quickgelu'
]  # noqa: E501


class CLIPScoreModel(ScoreModel):
    "A wrapper for OpenCLIP models (including openAI's CLIP, OpenCLIP, DatacompCLIP)"

    def __init__(self, model_name='openai:ViT-L-14', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in CLIP_MODELS
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        import open_clip

        from ..utils import download_open_clip_model

        self.pretrained, self.arch = self.model_name.split(':')
        # load model from modelscope
        model_file_path = download_open_clip_model(self.arch, self.pretrained, self.cache_dir)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.arch, pretrained=model_file_path, device=self.device)
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
