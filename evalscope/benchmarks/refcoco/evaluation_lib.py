from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO


def coco_score_result(results, doc):
    score_json = {}
    dataset = {'annotations': [], 'images': []}
    stored_results = []
    idx = 0
    ann_id = 0
    for result in results:
        stored_results.append({'image_id': idx, 'caption': result})
        for s in doc['answer']:
            dataset['annotations'].append({'image_id': idx, 'caption': s, 'id': ann_id})
            ann_id += 1

        dataset['images'].append({'id': idx})
        idx += 1

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params['image_id']
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Bleu(1-4)
    score, scores = Bleu(4).compute_score(gts, res)
    score_json['Bleu_1'] = score[0]
    score_json['Bleu_2'] = score[1]
    score_json['Bleu_3'] = score[2]
    score_json['Bleu_4'] = score[3]

    # METEOR
    score, scores = Meteor().compute_score(gts, res)
    score_json['METEOR'] = score

    # ROUGE_L
    score, scores = Rouge().compute_score(gts, res)
    score_json['ROUGE_L'] = score

    # CIDEr
    score, scores = Cider().compute_score(gts, res)
    score_json['CIDEr'] = score

    return score_json


def bbox_rec_score_result(results_list, doc):
    score_json = {}
    box1 = results_list
    box2 = doc['bbox']

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0

    score_json['IoU'] = iou
    score_json['ACC@0.1'] = (iou >= 0.1)
    score_json['ACC@0.3'] = (iou >= 0.3)
    score_json['ACC@0.5'] = (iou >= 0.5)
    score_json['ACC@0.7'] = (iou >= 0.7)
    score_json['ACC@0.9'] = (iou >= 0.9)

    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    score_json['Center_ACC'] = (box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3])

    return score_json


def process_results(doc, results):
    response = {}
    eval_mode = doc['eval_mode']
    results_list = results if isinstance(results, list) else [results]
    if eval_mode in ['bbox', 'seg']:
        response = coco_score_result(results_list, doc)
    elif eval_mode == 'bbox_rec':
        response = bbox_rec_score_result(results_list, doc)

    return response
