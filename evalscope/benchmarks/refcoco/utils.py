from PIL import ImageDraw


def refcoco_bbox_doc_to_visual(original_image, bbox):
    image = original_image.convert('RGB')
    draw = ImageDraw.Draw(image)
    bbox_xy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    draw.rectangle(bbox_xy, outline='red', width=3)
    return image.convert('RGB')


def refcoco_seg_doc_to_visual(original_image, segmentation):
    image = original_image.convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.polygon(segmentation, outline='red', width=3)
    return image.convert('RGB')
