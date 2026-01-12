import os
import json

import cv2
import numpy as np
from matplotlib import pyplot as plt

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from dataset import Adapter
from xyxy_conv import binary_mask_to_xyxy

MODEL_CHECKPOINT = "/Users/piotrswiecik/dev/ives/coronary/trained_models/detectron/20251208_094624_lr00025_freeze0/model_final.pth"
IMAGE_PATH = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/val/images/1.png"
SCORE_THRESHOLD = 0.2


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
    """Load ground truth annotations."""
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
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
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


if __name__ == "__main__":
    split, dataset_root, image_filename, image_id = infer_dataset_paths(IMAGE_PATH)
    adapter, raw_image_info, raw_annotations = load_ground_truth(dataset_root, split, image_id)
    num_classes = len(adapter.class_names)

    im = cv2.imread(IMAGE_PATH)

    instances = predict(im)

    poly_masks = instances["pred_masks"]
    coco_masks = []
    for mask in poly_masks:
        coco_mask = binary_mask_to_xyxy(mask)
        coco_masks.append(coco_mask)

    original = im[:, :, ::-1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(original)
    axes[0, 1].set_title("Predicted Masks", fontsize=14)
    axes[0, 1].axis("off")
    for coco_mask in coco_masks:
        for poly in coco_mask:
            if not poly:
                continue
            pts = np.array(poly, dtype=float).reshape(-1, 2)
            axes[0, 1].plot(pts[:, 0], pts[:, 1], "-r", linewidth=1.5)

    plt.tight_layout()
    plt.show()