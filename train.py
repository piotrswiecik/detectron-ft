import copy
import os
import json
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from typer import Typer
import torch

from dataset import Adapter

setup_logger()
app = Typer()


def custom_mapper(dataset_dict):
    """Applies augmentation and preprocessing."""
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    augmentations = [
        T.ResizeShortestEdge(
            short_edge_length=[640, 672, 704, 736, 768, 800],
            max_size=1333,
            sample_style="choice",
        ),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomRotation(angle=[-45, 45]),  # Rotate +/- 45 degrees
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
    ]
    aug_input = T.AugInput(image)
    transforms = T.AugmentationList(augmentations)(aug_input)
    image = aug_input.image
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


class ArcadeTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Hook in the custom_mapper defined above
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # Use COCOEvaluator to get AP/AP50 metrics
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


@app.command()
def train(data_path: str, out_dir: str, epochs: int, batch: int = 2, base_lr: float = 0.001):
    os.makedirs(out_dir, exist_ok=True)

    class_names = []
    num_train_images = 0
    splits = ["train", "val"]
    master_id_map = None

    for split in splits:
        json_file = os.path.join(data_path, split, "annotations", f"{split}.json")
        img_dir = os.path.join(data_path, split, "images")

        if not os.path.exists(json_file):
            continue

        with open(json_file) as f:
            raw_data = json.load(f)

        adapter = Adapter(raw_data, img_dir)

        if split == "train":
            class_names = adapter.class_names
            num_train_images = len(adapter.as_list())
            master_id_map = adapter.id_map
            print(
                f"Training data loaded: {num_train_images} images, classes: {class_names}"
            )

        d_name = f"arcade_{split}"
        DatasetCatalog.register(d_name, lambda a=adapter: a.as_list())
        MetadataCatalog.get(d_name).set(thing_classes=adapter.class_names)

    if num_train_images == 0:
        raise ValueError("No training images found")

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.DATASETS.TRAIN = ("arcade_train",)
    # Only add validation if we actually registered it
    cfg.DATASETS.TEST = ("arcade_val",) if "arcade_val" in DatasetCatalog.list() else ()

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

    # --- C. Solver Math ---
    one_epoch_iters = num_train_images // batch
    max_iter = one_epoch_iters * epochs

    cfg.SOLVER.IMS_PER_BATCH = batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.8))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = one_epoch_iters * 5
    cfg.TEST.EVAL_PERIOD = one_epoch_iters * 5

    cfg.OUTPUT_DIR = out_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = ArcadeTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    app()
