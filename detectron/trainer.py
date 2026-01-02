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
from detectron.losses import calculate_dice_score, calculate_iou_score

setup_logger()
load_dotenv()


def validation_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    augmentations = [
        T.ResizeShortestEdge(
            short_edge_length=[800],
            max_size=1333,
            sample_style="choice",
        )
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
            mapper=validation_mapper,
        )
        self.log = logging.getLogger(__name__)

    def _do_loss_eval(self):
        total = len(self.data_loader)
        num_warmup = min(5, total - 1)

        total_compute_time = 0
        losses = []
        iou_scores = []
        dice_scores = []

        # Debug counters
        total_samples = 0
        no_pred_instances = 0
        no_gt_instances = 0
        no_pred_masks = 0
        no_gt_masks = 0
        successful_calculations = 0
        empty_after_convert = 0
        zero_matches = 0
        non_finite_scores = 0
        exceptions = 0

        model = self.trainer.model  # type: ignore

        for idx, inputs in enumerate(self.data_loader):
            if idx == num_warmup:
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

                was_training = model.training
                model.eval()
                predictions = model(inputs)
                if was_training:
                    model.train()

                # Calculate IoU and DICE
                for input_dict, pred in zip(inputs, predictions):
                    total_samples += 1

                    if "instances" not in pred or len(pred["instances"]) == 0:
                        continue
                    if "instances" not in input_dict or len(input_dict["instances"]) == 0:
                        continue
                    if not pred["instances"].has("pred_masks"):
                        continue
                    if not input_dict["instances"].has("gt_masks"):
                        continue

                    pred_instances = pred["instances"]
                    gt_instances = input_dict["instances"]

                    if total_samples == 1:
                        with open("/tmp/iou_debug.txt", "w") as f:
                            f.write(f"Reached matching code\n")
                            f.write(f"Pred instances: {len(pred_instances)}\n")
                            f.write(f"GT instances: {len(gt_instances)}\n")
                            f.write(f"Pred has masks: {pred_instances.has('pred_masks')}\n")
                            f.write(f"GT has masks: {gt_instances.has('gt_masks')}\n")

                    # Convert GT masks
                    if hasattr(gt_instances.gt_masks, 'tensor'):
                        gt_mask_tensor = gt_instances.gt_masks.tensor
                    elif hasattr(gt_instances.gt_masks, 'to_bitmasks'):
                        h, w = gt_instances.image_size
                        gt_mask_tensor = gt_instances.gt_masks.to_bitmasks(h, w).tensor
                    else:
                        continue

                    if len(gt_mask_tensor) == 0:
                        continue

                    # Match pred masks to GT masks using box IoU
                    from detectron2.structures import pairwise_iou
                    box_iou = pairwise_iou(pred_instances.pred_boxes, gt_instances.gt_boxes)

                    if total_samples == 1:
                        with open("/tmp/iou_debug.txt", "a") as f:
                            f.write(f"Box IoU shape: {box_iou.shape}\n")
                            if box_iou.shape[0] > 0:
                                f.write(f"Box IoU max per GT: {box_iou.max(dim=0).values}\n")

                    # For each GT, find best matching prediction
                    matches_found = 0
                    for gt_idx in range(len(gt_instances)):
                        if box_iou.shape[0] == 0:
                            continue

                        # Find prediction with highest box IoU for this GT
                        best_pred_idx = box_iou[:, gt_idx].argmax()
                        best_iou = box_iou[best_pred_idx, gt_idx].item()

                        if total_samples == 1:
                            with open("/tmp/iou_debug.txt", "a") as f:
                                f.write(f"GT {gt_idx}: best_iou={best_iou:.3f}, threshold=0.5, pass={best_iou >= 0.5}\n")

                        # Only consider if box IoU > 0.5
                        if best_iou < 0.5:
                            continue

                        matches_found += 1

                        # Get masks
                        pred_mask = pred_instances.pred_masks[best_pred_idx]
                        gt_mask = gt_mask_tensor[gt_idx]

                        # Resize if needed
                        pred_mask = pred_mask.float().to(gt_mask.device)
                        gt_mask = gt_mask.float()

                        if pred_mask.shape != gt_mask.shape:
                            pred_mask = torch.nn.functional.interpolate(
                                pred_mask.unsqueeze(0).unsqueeze(0),
                                size=gt_mask.shape,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()

                        # Calculate metrics for this pair
                        iou = calculate_iou_score(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
                        dice = calculate_dice_score(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))

                        if total_samples == 1 and successful_calculations == 0:
                            with open("/tmp/iou_debug.txt", "a") as f:
                                f.write(f"FIRST CALCULATION:\n")
                                f.write(f"  IoU: {iou.item():.4f}\n")
                                f.write(f"  DICE: {dice.item():.4f}\n")
                                f.write(f"  Pred mask shape: {pred_mask.shape}\n")
                                f.write(f"  GT mask shape: {gt_mask.shape}\n")

                        iou_scores.append(iou.item())
                        dice_scores.append(dice.item())
                        successful_calculations += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

        # Calculate mean metrics
        mean_loss = torch.tensor(losses).mean().item()
        mean_iou = torch.tensor(iou_scores).mean().item() if iou_scores else 0.0
        mean_dice = torch.tensor(dice_scores).mean().item() if dice_scores else 0.0

        # Debug output
        print("\n" + "=" * 80)
        print(f"VALIDATION @ iter {self.trainer.iter}")
        print(f"Samples: {total_samples} | Success: {successful_calculations}")
        print(f"No pred_inst: {no_pred_instances} | No GT_inst: {no_gt_instances}")
        print(f"No pred_masks: {no_pred_masks} | No GT_masks: {no_gt_masks}")
        print(f"Empty after convert: {empty_after_convert} | Zero matches: {zero_matches}")
        print(f"Non-finite scores: {non_finite_scores} | Exceptions: {exceptions}")
        if iou_scores:
            print(f"IoU: {mean_iou:.4f} | DICE: {mean_dice:.4f}")
        else:
            print("NO SCORES - see counters above to diagnose")
        print("=" * 80 + "\n")

        # Store metrics in trainer storage
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        self.trainer.storage.put_scalar("validation_iou", mean_iou)
        self.trainer.storage.put_scalar("validation_dice", mean_dice)

        # Log metrics directly to MLflow
        mlflow.log_metrics(
            {
                "validation_loss": mean_loss,
                "validation_iou": mean_iou,
                "validation_dice": mean_dice,
            },
            step=self.trainer.iter,
        )

        self.log.info(f"Validation Loss: {mean_loss:.4f}, IoU: {mean_iou:.4f}, DICE: {mean_dice:.4f}")
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
