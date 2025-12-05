import copy
import logging
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
from detectron2.engine.hooks import HookBase
import mlflow
import torch
from dotenv import load_dotenv

from dataset import Adapter

setup_logger()
load_dotenv()


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    augmentations = [
        T.ResizeShortestEdge(
            short_edge_length=[640, 672, 704, 736, 768, 800],
            max_size=1333,
            sample_style="choice",
        ),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomRotation(angle=[-45, 45]),
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


class MLFlowHook(HookBase):
    def __init__(self, cfg, period: int = 20):
        self.cfg = cfg
        self.period = period

    def before_train(self):
        self._log_params_from_cfg(self.cfg)

    def after_step(self):
        if self.trainer.iter % self.period == 0:
            storage = self.trainer.storage
            metrics = {}
            for k, v in storage.latest().items():
                metrics[k] = v.median(self.period)
            mlflow.log_metrics(metrics, step=self.trainer.iter)

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            mlflow.log_artifacts(self.cfg.OUTPUT_DIR, artifact_path="model_output")

    @staticmethod
    def _log_params_from_cfg(cfg):
        params = {"SOLVER.BASE_LR": cfg.SOLVER.BASE_LR, "SOLVER.MAX_ITER": cfg.SOLVER.MAX_ITER,
                  "MODEL.ROI_HEADS.BATCH_SIZE": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
                  "DATASETS.TRAIN": str(cfg.DATASETS.TRAIN)}
        mlflow.log_params(params)


class ArcadeTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()

        hooks.append(MLFlowHook(self.cfg))
        return hooks


class ArcadeOrchestrator:
    def __init__(self, arcade_syntax_root: str, model_output_dir: str):
        self.log = logging.getLogger(__name__ + ".ArcadeOrchestrator")
        self.model_output_dir = model_output_dir

        os.makedirs(model_output_dir, exist_ok=True)

        self.class_names = []
        self.num_train_images = 0
        splits = ["train", "val"]

        for split in splits:
            json_file = os.path.join(arcade_syntax_root, split, "annotations", f"{split}.json")
            img_dir = os.path.join(arcade_syntax_root, split, "images")

            if not os.path.exists(json_file):
                continue

            with open(json_file) as f:
                raw_data = json.load(f)

            adapter = Adapter(raw_data, img_dir)

            if split == "train":
                self.class_names = adapter.class_names
                self.num_train_images = len(adapter.as_list())
                print(
                    f"Training data loaded: {self.num_train_images} images, classes: {self.class_names}"
                )

            d_name = f"arcade_{split}"
            DatasetCatalog.register(d_name, lambda a=adapter: a.as_list())
            MetadataCatalog.get(d_name).set(thing_classes=adapter.class_names)

            if self.num_train_images == 0:
                raise ValueError("No training images found")

            self.cfg = get_cfg()
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )

            self.cfg.DATASETS.TRAIN = ("arcade_train",)
            self.cfg.DATASETS.TEST = (
                ("arcade_val",) if "arcade_val" in DatasetCatalog.list() else ()
            )

            self.cfg.DATALOADER.NUM_WORKERS = 4
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )

            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_names)

            self.cfg.OUTPUT_DIR = self.model_output_dir
            os.makedirs(model_output_dir, exist_ok=True)

    def train(self, epochs: int, batch: int = 2, base_lr: float = 0.001):
        one_epoch_iters = self.num_train_images // batch
        max_iter = one_epoch_iters * epochs

        self.cfg.SOLVER.IMS_PER_BATCH = batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.8))
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = one_epoch_iters * 5
        self.cfg.TEST.EVAL_PERIOD = one_epoch_iters * 5

        experiment_name = os.getenv("MLFLOW_EXPERIMENT")
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI")
        )
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"run_epochs_{epochs}_batch_{batch}"):
            trainer = ArcadeTrainer(self.cfg)
            trainer.resume_or_load(resume=False)

            self.log.info(f"Starting training for {epochs} epochs")
            trainer.train()




