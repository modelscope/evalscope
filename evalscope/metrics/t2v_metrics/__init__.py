from __future__ import absolute_import, division, print_function

from .clipscore import CLIPScore, list_all_clipscore_models
from .constants import CACHE_DIR
from .itmscore import ITMScore, list_all_itmscore_models
from .vqascore import VQAScore, list_all_vqascore_models


def list_all_models():
    return list_all_vqascore_models() + list_all_clipscore_models() + list_all_itmscore_models()


def get_score_model(model='clip-flant5-xxl', device='cuda', cache_dir=CACHE_DIR, **kwargs):
    if model in list_all_vqascore_models():
        return VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)
    elif model in list_all_clipscore_models():
        return CLIPScore(model, device=device, cache_dir=cache_dir, **kwargs)
    elif model in list_all_itmscore_models():
        return ITMScore(model, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()


def clip_flant5_score():
    clip_flant5_score = VQAScore(model='clip-flant5-xxl')
    return clip_flant5_score


def pick_score():
    pick_score = CLIPScore(model='pickscore-v1')
    return pick_score


def clip_score():
    clip_score = CLIPScore(model='openai:ViT-L-14-336')
    return clip_score


def blip2_score():
    blip_itm_score = ITMScore(model='blip2-itm')
    return blip_itm_score


def hpsv2_score():
    hpsv2_score = CLIPScore(model='hpsv2')
    return hpsv2_score


def hpsv2_1_score():
    hpsv2_1_score = CLIPScore(model='hpsv2.1')
    return hpsv2_1_score


def image_reward_score():
    image_reward_score = ITMScore(model='image-reward-v1')
    return image_reward_score


def fga_blip2_score():
    fga_blip2_score = ITMScore(model='fga_blip2')
    return fga_blip2_score


def mps_score():
    mps_score = CLIPScore(model='mps')
    return mps_score
