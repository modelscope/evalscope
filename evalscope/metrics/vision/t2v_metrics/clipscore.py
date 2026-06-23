from typing import List

from .constants import CACHE_DIR
from .models.clipscore_models import get_clipscore_model, list_all_clipscore_models
from .score import Score


class CLIPScore(Score):

    def prepare_scoremodel(self, model='openai:ViT-L/14', device='cuda', cache_dir=CACHE_DIR):
        return get_clipscore_model(model, device=device, cache_dir=cache_dir)

    def list_all_models(self) -> List[str]:
        return list_all_clipscore_models()
