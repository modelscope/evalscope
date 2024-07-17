import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore

def fid_score(imgs_tensor1, imgs_tensor2):
    fid = FrechetInceptionDistance(feature=64)

    fid.update(imgs_tensor1, real=True)
    fid.update(imgs_tensor2, real=False)
    result = fid.compute().item()

    return result

def is_score(imgs_tensor):
    inception = InceptionScore()

    inception.update(imgs_tensor)
    result = inception.compute()
    is_mean, is_std = result[0].item(), result[1].item()

    return is_mean, is_std

def clip_score(img_tensor, prompt):
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    score = metric(img_tensor, prompt)
    result = score.detach()

    return result