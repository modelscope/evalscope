from ...constants import CACHE_DIR
from .clip_model import CLIP_MODELS, CLIPScoreModel
from .hpsv2_model import HPSV2_MODELS, HPSV2ScoreModel
from .mps_model import MPS_MODELS, MPSModel
from .pickscore_model import PICKSCORE_MODELS, PickScoreModel

ALL_CLIP_MODELS = [
    CLIP_MODELS,
    HPSV2_MODELS,
    PICKSCORE_MODELS,
    MPS_MODELS,
]


def list_all_clipscore_models():
    return [model for models in ALL_CLIP_MODELS for model in models]


def get_clipscore_model(model_name, device='cuda', cache_dir=CACHE_DIR):
    assert model_name in list_all_clipscore_models()
    if model_name in CLIP_MODELS:
        return CLIPScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in HPSV2_MODELS:
        return HPSV2ScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in PICKSCORE_MODELS:
        return PickScoreModel(model_name, device=device, cache_dir=cache_dir)
    elif model_name in MPS_MODELS:
        return MPSModel(model_name, device=device, cache_dir=cache_dir)
    else:
        raise NotImplementedError()
