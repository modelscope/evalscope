import os
import torch
from modelscope import AutoTokenizer
from typing import List, Union

from ...constants import CACHE_DIR
from ..model import ScoreModel
from ..vqascore_models.lavis.models import load_model_and_preprocess

FGA_BLIP2_MODELS = {
    'fga_blip2': {
        'variant': 'coco'
    },
}


def get_index(list1, list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0


class FGA_BLIP2ScoreModel(ScoreModel):
    'A wrapper for FGA BLIP-2 ITMScore models'

    def __init__(self, model_name='fga_blip2', device='cuda', cache_dir=CACHE_DIR):
        assert model_name in FGA_BLIP2_MODELS, f'Model name must be one of {FGA_BLIP2_MODELS.keys()}'
        os.environ['TORCH_HOME'] = cache_dir
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        from ..utils import download_file

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('AI-ModelScope/bert-base-uncased', truncation_side='right')
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        # load model
        self.variant = FGA_BLIP2_MODELS[self.model_name]['variant']
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            'fga_blip2', self.variant, is_eval=True, device=self.device)
        # load pretrained weights
        model_weight_path = download_file(
            'AI-ModelScope/FGA-BLIP2', file_name='fga_blip2.pth', cache_dir=self.cache_dir)
        self.model.load_checkpoint(model_weight_path)
        self.model.eval()

    def load_images(self, image):
        pass

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, images: List[str], texts: List[Union[str, dict]]) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), 'Number of images and texts must match'

        result_list = []
        for image_path, text in zip(images, texts):
            if isinstance(text, str):
                elements = []  # elements scores
                prompt = text
            else:
                elements = text['tags']
                prompt = text['prompt']

            image = self.image_loader(image_path)
            image = self.vis_processors['eval'](image).to(self.device)
            prompt = self.text_processors['eval'](prompt)
            prompt_ids = self.tokenizer(prompt).input_ids

            alignment_score, scores = self.model.element_score(image.unsqueeze(0), [prompt])

            elements_score = dict()
            for element in elements:
                element_ = element.rpartition('(')[0]
                element_ids = self.tokenizer(element_).input_ids[1:-1]

                idx = get_index(element_ids, prompt_ids)
                if idx:
                    mask = [0] * len(prompt_ids)
                    mask[idx:idx + len(element_ids)] = [1] * len(element_ids)

                    mask = torch.tensor(mask).to(self.device)
                    elements_score[element] = (scores * mask).sum() / mask.sum()
                else:
                    elements_score[element] = torch.tensor(0.0).to(self.device)
            if elements_score:
                result_list.append({'overall_score': alignment_score, **elements_score})
            else:
                result_list.append(alignment_score)

        return result_list
