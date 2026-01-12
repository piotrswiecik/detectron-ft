"""
Simple script to generate predictions on a single image.
Uses EXACT same logic as sample.py.
"""
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from dotenv import load_dotenv

from dataset import Adapter

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CHECKPOINT = "/Users/piotrswiecik/dev/ives/coronary/trained_models/detectron/20251208_094624_lr00025_freeze0/model_final.pth"
IMAGE_PATH = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/test/images/1.png"
SCORE_THRESHOLD = 0.2

# ============================================================================


def infer_dataset_paths(image_path):
    """Infer ARCADE dataset structure from image path."""
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


def visualize_predictions(image, instances, raw_annotations, image_height, image_width, class_names):
    """
    Create 4-panel visualization.
    """
    # Panel 1: Original image (convert BGR to RGB for matplotlib)
    original = image[:, :, ::-1]

    # Panel 2: Detectron2 visualization (EXACT same as sample.py)
    temp_metadata = MetadataCatalog.get("temp_viz")
    temp_metadata.set(thing_classes=class_names)
    v = Visualizer(
        image[:, :, ::-1], metadata=temp_metadata, scale=1.0, instance_mode=ColorMode.IMAGE
    )
    out = v.draw_instance_predictions(instances.to("cpu"))
    detectron_result = out.get_image()

    # Panel 3: Ground truth overlay
    gt_overlay = image[:, :, ::-1].copy()
    for ann in raw_annotations:
        seg = ann['segmentation']
        if isinstance(seg, list) and len(seg) >= 6:
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.polylines(gt_overlay, [pts], True, (0, 255, 0), 2)

    # Panel 4: Prediction masks overlay
    pred_overlay = image[:, :, ::-1].copy()
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()
        for i in range(len(masks)):
            mask = masks[i]
            pred_overlay[mask] = pred_overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(detectron_result)
    axes[0, 1].set_title(f"Detectron2 Visualization ({len(instances)} detections)", fontsize=14)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(gt_overlay)
    axes[1, 0].set_title(f"Ground Truth ({len(raw_annotations)} annotations)", fontsize=14)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title("Prediction Masks (Red)", fontsize=14)
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print("="*70)
    print("SINGLE IMAGE PREDICTION - USING sample.py LOGIC")
    print("="*70)

    # Check paths
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"ERROR: Model not found: {MODEL_CHECKPOINT}")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        return

    # Parse dataset structure
    print("\nParsing dataset structure...")
    split, dataset_root, image_filename, image_id = infer_dataset_paths(IMAGE_PATH)
    print(f"Image ID: {image_id}")

    # Load ground truth
    print("\nLoading ground truth...")
    adapter, raw_image_info, raw_annotations = load_ground_truth(dataset_root, split, image_id)
    num_classes = len(adapter.class_names)
    print(f"Ground truth annotations: {len(raw_annotations)}")

    # Setup config (EXACT same as sample.py)
    print("\nLoading model...")
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = MODEL_CHECKPOINT
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)
    print("Model loaded!")

    # Load image and run inference (EXACT same as sample.py)
    print("\nRunning inference...")
    im = cv2.imread(IMAGE_PATH)
    if im is None:
        print("ERROR: Failed to read image")
        return

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    print(f"Detected instances: {len(instances)}")
    if len(instances) > 0:
        print(f"Score range: {instances.scores.min():.4f} - {instances.scores.max():.4f}")
        if instances.has("pred_masks"):
            print(f"Pred masks shape: {instances.pred_masks.shape}")

    # Visualize
    print("\nCreating visualization...")
    visualize_predictions(
        im,
        instances,
        raw_annotations,
        raw_image_info['height'],
        raw_image_info['width'],
        adapter.class_names
    )


if __name__ == "__main__":
    main()
