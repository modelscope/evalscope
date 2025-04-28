'''
 * Adapted from BLIP (https://github.com/salesforce/BLIP)
'''

from modelscope import AutoTokenizer
from torch import nn

from ...vqascore_models.lavis.models.med import BertConfig, BertModel
from ...vqascore_models.lavis.models.vit import VisionTransformer


def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('AI-ModelScope/bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):

    assert vit in ['base', 'large'], 'vit parameter must be base or large'
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate)
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate)
    return visual_encoder, vision_width


class BLIP_Pretrain(nn.Module):

    def __init__(
        self,
        med_config='med_config.json',
        image_size=224,
        vit='base',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        queue_size=57600,
        momentum=0.995,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        encoder_config.add_type_embeddings = False
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
