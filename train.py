import os
import json
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from typer import Typer

from dataset import Adapter

setup_logger()
app = Typer()


@app.command()
def train(path: str, out: str):
    annotation_file = os.path.join(path, "train", "annotations", "train.json")
    image_dir = os.path.join(path, "train", "images",
)
    os.makedirs(out, exist_ok=True)

    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file, "r") as f:
        labels = json.load(f)

    adapter = Adapter(labels, image_dir)

    dataset_name = "arcade_custom_train"
    DatasetCatalog.register(dataset_name, lambda: adapter.as_list())

    MetadataCatalog.get(dataset_name).set(thing_classes=adapter.class_names)

    print(f"Registered {dataset_name} with classes: {adapter.class_names}")

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(adapter.class_names)

    cfg.OUTPUT_DIR = out
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    app()
