from ...constants import CACHE_DIR
from .blip2_itm_model import BLIP2_ITM_MODELS, BLIP2ITMScoreModel
from .fga_blip2_model import FGA_BLIP2_MODELS, FGA_BLIP2ScoreModel
from .image_reward_model import IMAGE_REWARD_MODELS, ImageRewardScoreModel

ALL_ITM_MODELS = [
    BLIP2_ITM_MODELS,
    IMAGE_REWARD_MODELS,
    FGA_BLIP2_MODELS,
]


def list_all_itmscore_models():
    return [model for models in ALL_ITM_MODELS for model in models]


def get_itmscore_model(model_name, device='cuda', cache_dir=CACHE_DIR):
    assert model_name in list_all_itmscore_models()
    if model_name in BLIP2_ITM_MODELS:
        return BLIP2ITMScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in IMAGE_REWARD_MODELS:
        return ImageRewardScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in FGA_BLIP2_MODELS:
        return FGA_BLIP2ScoreModel(model_name, device=device, cache_dir=cache_dir)
    else:
        raise NotImplementedError()
