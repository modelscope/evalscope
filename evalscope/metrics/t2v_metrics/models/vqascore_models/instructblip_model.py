import copy
import os
import torch
from torchvision import transforms
from typing import List

from ...constants import CACHE_DIR
from .lavis.models import load_model
from .vqa_model import VQAScoreModel

default_question_template = 'Question: Does this figure show "{}"? Please answer yes or no.'
default_answer_template = 'yes'  # instruct-blip uses "yes" instead of "Yes"

InstructBLIP_MODELS = {
    'instructblip-flant5-xxl': {
        'variant': 'flant5xxl'
    },
    'instructblip-flant5-xl': {
        'variant': 'flant5xl'
    },
}


class InstructBLIPModel(VQAScoreModel):
    """A wrapper for the InstructBLIP (FlanT5-based) models"""

    def __init__(self, model_name='instructblip-flant5-xxl', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in InstructBLIP_MODELS, f'Model name {model_name} not found in InstructBLIP_MODELS'
        os.environ['TORCH_HOME'] = cache_dir
        import timm.models.hub as timm_hub

        # if cache_dir != timm_hub.get_cache_dir():
        #     print(f"Warning: cache_dir {cache_dir} won't be used because "
        #            "InstructBLIP is cached using timm_hub.get_cache_dir(). "
        #            "You may want to update os.environ['TORCH_HOME'] before importing to specify the cache_dir.")
        #     print(f"Model saved to {timm_hub.get_cache_dir()}")
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        self.variant = InstructBLIP_MODELS[self.model_name]['variant']
        self.model = load_model('blip2_t5_instruct', self.variant, is_eval=True, device=self.device)
        size = 224
        self.image_preprocess = transforms.Compose([
            transforms.Resize((size), interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop((size, size)),
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
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str = default_question_template,
                answer_template: str = default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), 'Number of images and texts must match'
        # Turn "a photo of a dog" into
        # Q: "Question: Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        images = self.load_images(images)

        image_feat = self.model.ln_vision(self.model.visual_encoder(images))
        image_att = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(self.device)
        query_token = self.model.query_tokens.expand(image_feat.shape[0], -1, -1)
        query_att = torch.ones(query_token.size()[:-1], dtype=torch.long).to(query_token.device)

        question_text_Qformer = self.model.tokenizer(
            questions,
            padding='longest',
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors='pt',
        ).to(query_token.device)
        Qformer_atts = torch.cat([query_att, question_text_Qformer.attention_mask], dim=1)

        query_output = self.model.Qformer.bert(
            question_text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_token,
            encoder_hidden_states=image_feat,
            encoder_attention_mask=image_att,
            return_dict=True,
        )

        t5_input = self.model.t5_proj(query_output.last_hidden_state[:, :query_token.size(1), :])
        t5_att = torch.ones(t5_input.size()[:-1], dtype=torch.long).to(query_token.device)

        question_text_t5 = self.model.t5_tokenizer(
            questions,
            padding='longest',
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors='pt',
        ).to(query_token.device)
        answer_text_t5 = self.model.t5_output_tokenizer(
            answers,
            padding='longest',
            truncation=True,
            max_length=self.model.max_output_txt_len,
            return_tensors='pt',
        ).to(query_token.device)

        encoder_atts = torch.cat([t5_att, question_text_t5.attention_mask], dim=1)

        labels = answer_text_t5.input_ids.masked_fill(answer_text_t5.input_ids == self.model.t5_tokenizer.pad_token_id,
                                                      -100)

        inputs_embeds = self.model.t5_model.encoder.embed_tokens(question_text_t5.input_ids)
        inputs_embeds = torch.cat([t5_input, inputs_embeds], dim=1)
        outputs = self.model.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=answer_text_t5.attention_mask,
            return_dict=True,
            labels=labels,
        )

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        lm_prob = torch.zeros(outputs.logits.shape[0])
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(outputs.logits[k], labels[k])).exp()
        return lm_prob
