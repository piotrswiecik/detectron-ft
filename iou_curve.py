import os

import cv2

from conv_utils import binary_mask_to_xyxy

MODEL_CHECKPOINT = "/Users/piotrswiecik/dev/ives/coronary/trained_models/detectron/20251208_094624_lr00025_freeze0/model_final.pth"
IMAGE_ROOT = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/val/images"
SCORE_THRESHOLD = 0.2
ANNOT_PATH = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/val/annotations/val.json"

from validate_iou import infer_dataset_paths, load_ground_truth, predict, calculate_iou, calculate_image_iou, \
    MappedGTAnnotation, ConvertedAnnotation


def parse_single(path, thr):
    split, dataset_root, image_filename = infer_dataset_paths(path)
    adapter, raw_image_info, raw_annotations, image_id = load_ground_truth(dataset_root, split, image_filename)

    mapped_gt_anns: list[MappedGTAnnotation] = []
    for img_record in adapter:
        if img_record["image_id"] == image_id:
            mapped_gt_anns = img_record["annotations"]
            break

    im = cv2.imread(path)
    instances = predict(im, thr=thr)

    conv_anns: list[ConvertedAnnotation] = []

    for cls, score, mask, box in zip(instances["pred_classes"], instances["scores"], instances["pred_masks"],
                                     instances["pred_boxes"]):
        coco_mask = binary_mask_to_xyxy(mask)
        conv_anns.append({
            "class_id": int(cls),
            "confidence": float(score),
            "mask": coco_mask,
            "box": box,  # TODO: convert later
        })
    image_iou = calculate_image_iou(conv_anns, mapped_gt_anns, im.shape)
    return image_iou


def single_pass(thr):
    ious = []
    cnt = 0
    for fn in os.listdir(IMAGE_ROOT):
        if fn.endswith(".png") or fn.endswith(".jpg"):
            cnt += 1
            IMAGE_PATH = os.path.join(IMAGE_ROOT, fn)
            img_iou = parse_single(IMAGE_PATH, thr=thr)
            ious.append(img_iou)

    avg_iou = sum(ious) / len(ious)
    min_iou = min(ious)
    max_iou = max(ious)

    return {
        "threshold": thr,
        "average_iou": avg_iou,
        "min_iou": min_iou,
        "max_iou": max_iou,
    }


if __name__ == "__main__":
    res = single_pass(thr=0.5)

