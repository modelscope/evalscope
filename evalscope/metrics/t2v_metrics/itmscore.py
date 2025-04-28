from typing import List

from .constants import CACHE_DIR
from .models.itmscore_models import get_itmscore_model, list_all_itmscore_models
from .score import Score


class ITMScore(Score):

    def prepare_scoremodel(self, model='blip2-itm', device='cuda', cache_dir=CACHE_DIR):
        return get_itmscore_model(model, device=device, cache_dir=cache_dir)

    def list_all_models(self) -> List[str]:
        return list_all_itmscore_models()
