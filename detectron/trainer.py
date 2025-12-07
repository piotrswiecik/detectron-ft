import copy
import datetime
import logging
import math
import os
import json
import time
import uuid

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm
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


class EvalHook(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.period = cfg.TEST.EVAL_PERIOD
        self.data_loader = build_detection_test_loader(
            cfg,
            cfg.DATASETS.TEST[0],
            mapper=None,
        )
        self.log = logging.getLogger(__name__)

    def _do_loss_eval(self):
        total = len(self.data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []

        model = self.trainer.model  # type: ignore

        for idx, inputs in enumerate(self.data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                if model.training:
                    loss_dict = model(inputs)
                else:
                    model.train()
                    loss_dict = model(inputs)
                    model.eval()

            losses.append(sum(loss for loss in loss_dict.values()))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

        mean_loss = torch.tensor(losses).mean().item()

        self.trainer.storage.put_scalar("validation_loss", mean_loss)

        self.log.info(f"Validation Loss: {mean_loss:.4f}")
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.period > 0 and next_iter % self.period == 0):
            self._do_loss_eval()


class MLFlowHook(HookBase):
    def __init__(self, cfg, log_period: int = 100):
        self.cfg = cfg
        self.log_period = log_period

    def before_train(self):
        self._log_params_from_cfg(self.cfg)

    def after_step(self):
        if self.trainer.iter % self.log_period == 0:
            storage = self.trainer.storage
            metrics = {}

            latest_keys = storage.latest().keys()

            for k in latest_keys:
                if k in storage.histories():
                    val = storage.histories()[k].median(self.log_period)
                    if math.isfinite(val):
                        metrics[k] = val
                    else:
                        metrics[k] = 0.0

            if metrics:
                mlflow.log_metrics(metrics, step=self.trainer.iter)

            mlflow.log_metrics(metrics, step=self.trainer.iter)

    def after_train(self):
        self.after_step()

    @staticmethod
    def _log_params_from_cfg(cfg):
        params = {
            "SOLVER.BASE_LR": cfg.SOLVER.BASE_LR,
            "SOLVER.MAX_ITER": cfg.SOLVER.MAX_ITER,
            "MODEL.ROI_HEADS.BATCH_SIZE": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "DATASETS.TRAIN": str(cfg.DATASETS.TRAIN),
        }
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
        hooks.append(EvalHook(self.cfg))
        hooks.append(MLFlowHook(self.cfg))
        return hooks


class ArcadeOrchestrator:
    def __init__(self, arcade_syntax_root: str, model_output_dir: str):
        self.log = logging.getLogger(__name__ + ".ArcadeOrchestrator")
        self.model_output_dir = model_output_dir

        self.class_names = []
        self.num_train_images = 0
        splits = ["train", "val"]

        for split in splits:
            json_file = os.path.join(
                arcade_syntax_root, split, "annotations", f"{split}.json"
            )
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

            DatasetCatalog.register(f"arcade_{split}", lambda a=adapter: a.as_list())
            MetadataCatalog.get(f"arcade_{split}").set(thing_classes=adapter.class_names)

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

    def train(
        self,
        epochs: int,
        batch: int = 2,
        base_lr: float = 0.00025,
        hyperparameters: dict | None = None,
    ):
        hyperparameters = hyperparameters or {}

        one_epoch_iters = self.num_train_images // batch
        max_iter = one_epoch_iters * epochs

        self.cfg.SOLVER.IMS_PER_BATCH = batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.WARMUP_ITERS = 1000  # ramp up learning rate
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.8))
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = one_epoch_iters * 5

        self.cfg.TEST.EVAL_PERIOD = one_epoch_iters * 5

        self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
        self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
        self.cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

        # backbone freeze (0, 1, 2)
        if "freeze_at" in hyperparameters and hyperparameters["freeze_at"] in [0, 1, 2]:
            self.cfg.MODEL.BACKBONE.FREEZE_AT = hyperparameters["freeze_at"]

        # anchor sizes
        if "anchor_sizes" in hyperparameters:
            sizes = hyperparameters["anchor_sizes"]
            if len(sizes) != 5:
                self.log.warning(
                    f"Warning: FPN expects 5 anchor sizes, got {len(sizes)}. This might crash."
                )
            self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = sizes

        # anchor aspect ratios
        if "anchor_ratios" in hyperparameters:
            self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = hyperparameters["anchor_ratios"]

        # roi head batch size
        if "roi_batch_size" in hyperparameters:
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = hyperparameters["roi_batch_size"]

        experiment_name = os.getenv("MLFLOW_EXPERIMENT")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"run_epochs_{epochs}_batch_{batch}"):
            trainer = ArcadeTrainer(self.cfg)
            trainer.resume_or_load(resume=False)

            self.log.info(f"Starting training for {epochs} epochs")
            trainer.train()
