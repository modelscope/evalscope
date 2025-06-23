def clip_flant5_score():
    from .vqascore import VQAScore
    clip_flant5_score = VQAScore(model='clip-flant5-xxl')
    return clip_flant5_score


def pick_score():
    from .clipscore import CLIPScore
    pick_score = CLIPScore(model='pickscore-v1')
    return pick_score


def clip_score():
    from .clipscore import CLIPScore
    clip_score = CLIPScore(model='openai:ViT-L-14-336')
    return clip_score


def blip2_score():
    from .itmscore import ITMScore
    blip_itm_score = ITMScore(model='blip2-itm')
    return blip_itm_score


def hpsv2_score():
    from .clipscore import CLIPScore
    hpsv2_score = CLIPScore(model='hpsv2')
    return hpsv2_score


def hpsv2_1_score():
    from .clipscore import CLIPScore
    hpsv2_1_score = CLIPScore(model='hpsv2.1')
    return hpsv2_1_score


def image_reward_score():
    from .itmscore import ITMScore
    image_reward_score = ITMScore(model='image-reward-v1')
    return image_reward_score


def fga_blip2_score():
    from .itmscore import ITMScore
    fga_blip2_score = ITMScore(model='fga_blip2')
    return fga_blip2_score


def mps_score():
    from .clipscore import CLIPScore
    mps_score = CLIPScore(model='mps')
    return mps_score
