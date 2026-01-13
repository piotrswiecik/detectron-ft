import os
import json
import typing

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from typing import TypedDict

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from dataset import Adapter
from conv_utils import binary_mask_to_xyxy, polygon_to_mask

MODEL_CHECKPOINT = "/Users/piotrswiecik/dev/ives/coronary/trained_models/detectron/20251208_094624_lr00025_freeze0/model_final.pth"
IMAGE_ROOT = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/val/images"
SCORE_THRESHOLD = 0.2
ANNOT_PATH = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/val/annotations/val.json"


class ConvertedAnnotation(TypedDict):
    class_id: int
    confidence: float
    mask: list[list[float]]
    box: typing.Any


class ArcadeAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    segmentation: list[list[float]]


class MappedGTAnnotation(TypedDict):
    bbox: list[float]
    bbox_mode: int
    category_id: int
    segmentation: list[list[float]]


def infer_dataset_paths(image_path):
    parts = image_path.split(os.sep)
    try:
        images_idx = parts.index('images')
        split = parts[images_idx - 1]
        dataset_root = os.sep.join(parts[:images_idx - 1])
        image_filename = parts[-1]
        image_id = int(os.path.splitext(image_filename)[0])
        return split, dataset_root, image_filename, image_id
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse ARCADE dataset structure from path: {image_path}")


def load_ground_truth(dataset_root, split, image_id):
    json_path = os.path.join(dataset_root, split, "annotations", f"{split}.json")
    img_dir = os.path.join(dataset_root, split, "images")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    with open(json_path, 'r') as f:
        arcade_data = json.load(f)

    adapter = Adapter(arcade_data, img_dir)

    raw_image_info = None
    for image in arcade_data.get("images", []):
        if image["id"] == image_id:
            raw_image_info = image
            break

    if raw_image_info is None:
        raise ValueError(f"Image ID {image_id} not found in {json_path}")

    raw_annotations = [
        ann for ann in arcade_data.get("annotations", [])
        if ann["image_id"] == image_id
    ]

    return adapter, raw_image_info, raw_annotations


def predict(im):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 25
    cfg.MODEL.WEIGHTS = MODEL_CHECKPOINT
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    instances = outputs["instances"]

    # We want to safely access all keys
    out = dict()

    if instances.has("pred_masks"):
        out["pred_masks"] = instances.pred_masks.to("cpu")
    else:
        out["pred_masks"] = None

    if instances.has("pred_boxes"):
        out["pred_boxes"] = instances.pred_boxes.to("cpu")
    else:
        out["pred_boxes"] = None

    if instances.has("scores"):
        out["scores"] = instances.scores.to("cpu")
    else:
        out["scores"] = None

    if instances.has("pred_classes"):
        out["pred_classes"] = instances.pred_classes.to("cpu")
    else:
        out["pred_classes"] = None

    return out


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_image_iou(conv_anns: list, mapped_gt_anns: list, image_shape: tuple) -> float:
    combined_pred_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    combined_gt_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for pred_ann in conv_anns:
        matching_gt = [ann for ann in mapped_gt_anns if ann["category_id"] == pred_ann["class_id"]]
        if not matching_gt:
            continue

        for poly in pred_ann["mask"]:
            if not poly:
                continue
            pred_mask = polygon_to_mask(poly, image_shape)
            combined_pred_mask = np.logical_or(combined_pred_mask, pred_mask).astype(np.uint8)

        for gt_ann in matching_gt:
            for poly in gt_ann["segmentation"]:
                if not poly:
                    continue
                gt_mask = polygon_to_mask(poly, image_shape)
                combined_gt_mask = np.logical_or(combined_gt_mask, gt_mask).astype(np.uint8)

    return calculate_iou(combined_pred_mask, combined_gt_mask)


def visualize_single(path):
    split, dataset_root, image_filename, image_id = infer_dataset_paths(path)
    adapter, raw_image_info, raw_annotations = load_ground_truth(dataset_root, split, image_id)

    raw_anns = [
        ArcadeAnnotation(id=ann["id"], image_id=ann["image_id"], category_id=ann["category_id"], segmentation=ann["segmentation"])
        for ann in raw_annotations
    ]

    mapped_gt_anns: list[MappedGTAnnotation] = []
    for img_record in adapter:
        if img_record["image_id"] == image_id:
            mapped_gt_anns = img_record["annotations"]
            break

    num_classes = len(adapter.class_names)

    im = cv2.imread(path)

    instances = predict(im)

    conv_anns: list[ConvertedAnnotation] = []

    for cls, score, mask, box in zip(instances["pred_classes"], instances["scores"], instances["pred_masks"], instances["pred_boxes"]):
        coco_mask = binary_mask_to_xyxy(mask)
        conv_anns.append({
            "class_id": int(cls),
            "confidence": float(score),
            "mask": coco_mask,
            "box": box, # TODO: convert later
        })

    original = im[:, :, ::-1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(original)
    axes[0, 1].set_title("Predicted Masks", fontsize=14)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(original)
    axes[1, 0].set_title("Ground Truth Masks", fontsize=14)
    axes[1, 0].axis("off")

    cmap = plt.cm.get_cmap("tab10", num_classes)
    class_colors = {i: cmap(i) for i in range(num_classes)}

    for coco_ann in conv_anns:
        matching_gt = list(filter(lambda ann: ann["category_id"] == coco_ann["class_id"], mapped_gt_anns))
        if not matching_gt:
            continue
        color = class_colors[coco_ann["class_id"]]
        for poly in coco_ann["mask"]:
            if not poly:
                continue
            pts = np.array(poly, dtype=float).reshape(-1, 2)
            axes[0, 1].plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=1.5)
        for gt_ann in matching_gt:
            for poly in gt_ann["segmentation"]:
                if not poly:
                    continue
                pts = np.array(poly, dtype=float).reshape(-1, 2)
                axes[1, 0].plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=1.5)

    image_iou = calculate_image_iou(conv_anns, mapped_gt_anns, im.shape)

    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5, f"Image IoU: {image_iou:.4f}", fontsize=20,
                    ha="center", va="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    IMAGE_PATH = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/val/images/16.png"
    visualize_single(IMAGE_PATH)
