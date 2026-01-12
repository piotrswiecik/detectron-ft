import copy
import datetime
import logging
import math
import os
import json
import time

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
import numpy as np

from dataset import Adapter
from converters import DetectronToArcadeConverter, ArcadeMetricsCalculator

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
        self.log = logging.getLogger(__name__)

        self.reverse_id_map = self._get_reverse_id_map()
        self.converter = DetectronToArcadeConverter(self.reverse_id_map)
        self.metrics_calculator = ArcadeMetricsCalculator()

        self.arcade_ground_truth = self._load_arcade_ground_truth()

    def _get_reverse_id_map(self):
        """Get reverse ID mapping from metadata."""
        try:
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
            if hasattr(metadata, 'id_reverse_map'):
                return metadata.id_reverse_map
        except (IndexError, KeyError):
            pass

        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return {i: i for i in range(num_classes)}

    def _load_arcade_ground_truth(self):
        """Load ARCADE ground truth annotations indexed by image_id."""
        try:
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        except (IndexError, KeyError):
            self.log.warning("No test dataset configured. ARCADE metrics unavailable.")
            return {}

        if hasattr(metadata, 'adapter_instance'):
            adapter = metadata.adapter_instance

            # Group annotations by image_id
            gt_by_image = {}
            for image in adapter.images:
                image_id = image["id"]
                gt_by_image[image_id] = {
                    "annotations": [
                        ann for ann in adapter._raw_anns
                        if ann["image_id"] == image_id
                    ],
                    "height": image["height"],
                    "width": image["width"]
                }

            return gt_by_image
        else:
            self.log.warning(
                "Adapter instance not found in metadata. "
                "ARCADE metrics will be unavailable. "
                "Ensure dataset registration includes adapter_instance=adapter"
            )
            return {}

    def _build_data_loader(self):
        """Build a fresh data loader for evaluation."""
        return build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            mapper=validation_mapper,
        )

    def _run_combined_evaluation(self):
        """
        Run both loss evaluation and ARCADE metrics in a SINGLE PASS.

        This avoids the DataLoader exhaustion bug where iterating twice
        over the same DataLoader produces no results on the second pass.
        """
        data_loader = self._build_data_loader()

        total = len(data_loader)
        num_warmup = min(5, total - 1)

        model = self.trainer.model
        was_training = model.training

        losses = []
        all_ious = []
        all_dices = []
        total_matches = 0
        total_predictions = 0
        total_ground_truth = 0

        self.converter.reset_counter()

        print(f"[DIAG] arcade_ground_truth has {len(self.arcade_ground_truth)} images")
        if self.arcade_ground_truth:
            sample_key = next(iter(self.arcade_ground_truth.keys()))
            sample_val = self.arcade_ground_truth[sample_key]
            print(f"[DIAG] Sample GT key type: {type(sample_key).__name__}, value: {sample_key}")
            print(f"[DIAG] Sample GT has {len(sample_val['annotations'])} annotations, dims: {sample_val['height']}x{sample_val['width']}")

        start_time = time.perf_counter()
        total_compute_time = 0

        diag_images_processed = 0
        diag_images_with_gt = 0
        diag_id_mismatches = 0
        diag_raw_instances_total = 0
        diag_score_filtered = 0

        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                model.train()
                loss_dict = model(inputs)
                losses.append(sum(loss for loss in loss_dict.values()))

                if self.arcade_ground_truth:
                    model.eval()
                    outputs = model(inputs)

                    for input_dict, output in zip(inputs, outputs):
                        diag_images_processed += 1
                        image_id = input_dict["image_id"]

                        if diag_images_processed == 1:
                            print(f"[DIAG] First input image_id: {image_id} (type: {type(image_id).__name__})")
                            print(f"[DIAG] First input image shape: {input_dict['image'].shape}")
                            instances = output["instances"].to("cpu")
                            print(f"[DIAG] First output has {len(instances)} raw instances")
                            if len(instances) > 0:
                                print(f"[DIAG] First instance scores: {instances.scores[:5].tolist()}")
                                if instances.has("pred_masks"):
                                    print(f"[DIAG] pred_masks shape: {instances.pred_masks.shape}")
                                    print(f"[DIAG] pred_masks dtype: {instances.pred_masks.dtype}")

                        if image_id not in self.arcade_ground_truth:
                            diag_id_mismatches += 1
                            if diag_id_mismatches <= 3:
                                print(f"[DIAG] image_id {image_id} not in arcade_ground_truth")
                            continue

                        diag_images_with_gt += 1
                        gt_data = self.arcade_ground_truth[image_id]
                        original_height = gt_data["height"]
                        original_width = gt_data["width"]

                        transformed_height = input_dict["image"].shape[1]
                        transformed_width = input_dict["image"].shape[2]

                        instances = output["instances"].to("cpu")
                        diag_raw_instances_total += len(instances)

                        if len(instances) > 0:
                            scores_above = (instances.scores >= 0.5).sum().item()
                            diag_score_filtered += (len(instances) - scores_above)

                        if len(instances) > 0 and instances.has("pred_masks"):
                            mask_height = instances.pred_masks.shape[1]
                            mask_width = instances.pred_masks.shape[2]
                        else:
                            mask_height = transformed_height
                            mask_width = transformed_width

                        pred_arcade = self.converter.convert_instances(
                            instances,
                            image_id,
                            original_height=original_height,
                            original_width=original_width,
                            transformed_height=mask_height,
                            transformed_width=mask_width,
                            score_threshold=0.5
                        )

                        gt_arcade = gt_data["annotations"]

                        if diag_images_with_gt == 1:
                            print(f"[DIAG] Mask dims: {mask_height}x{mask_width}, Original: {original_height}x{original_width}, Image tensor: {transformed_height}x{transformed_width}")
                            print(f"[DIAG] Scale factors (mask->original): x={original_width/mask_width:.3f}, y={original_height/mask_height:.3f}")
                            print(f"[DIAG] pred_arcade has {len(pred_arcade)} annotations after conversion")
                            print(f"[DIAG] gt_arcade has {len(gt_arcade)} annotations")
                            if pred_arcade:
                                print(f"[DIAG] First pred segmentation (first 10 coords): {pred_arcade[0]['segmentation'][:10]}")
                            if gt_arcade:
                                seg = gt_arcade[0].get('segmentation', [])
                                print(f"[DIAG] First GT segmentation (first 10 coords): {seg[:10] if seg else 'empty'}")

                        metrics = self.metrics_calculator.calculate_metrics_for_image(
                            pred_arcade,
                            gt_arcade,
                            original_height,
                            original_width,
                            iou_threshold=0.5
                        )

                        all_ious.extend(metrics["ious"])
                        all_dices.extend(metrics["dices"])
                        total_matches += metrics["num_matches"]
                        total_predictions += metrics["num_predictions"]
                        total_ground_truth += metrics["num_ground_truth"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

        print(f"[DIAG] Images processed: {diag_images_processed}, with GT: {diag_images_with_gt}, ID mismatches: {diag_id_mismatches}")
        print(f"[DIAG] Raw instances total: {diag_raw_instances_total}, filtered by score: {diag_score_filtered}")

        if was_training:
            model.train()
        else:
            model.eval()

        mean_loss = torch.tensor(losses).mean().item()
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        print(f"Validation Loss: {mean_loss:.4f}")

        if self.arcade_ground_truth:
            mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
            mean_dice = float(np.mean(all_dices)) if all_dices else 0.0
            precision = total_matches / total_predictions if total_predictions > 0 else 0.0
            recall = total_matches / total_ground_truth if total_ground_truth > 0 else 0.0

            self.trainer.storage.put_scalar("arcade/mean_iou", mean_iou)
            self.trainer.storage.put_scalar("arcade/mean_dice", mean_dice)
            self.trainer.storage.put_scalar("arcade/precision", precision)
            self.trainer.storage.put_scalar("arcade/recall", recall)

            print(
                f"ARCADE Metrics - IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f} "
                f"(matches: {total_matches}, preds: {total_predictions}, gt: {total_ground_truth})"
            )

        comm.synchronize()

    def after_step(self):
        """Called after each training step."""
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter

        if is_final or (self.period > 0 and next_iter % self.period == 0):
            self._run_combined_evaluation()


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

        # Log anchor-related parameters if available
        try:
            if hasattr(cfg.MODEL, "BACKBONE") and hasattr(cfg.MODEL.BACKBONE, "FREEZE_AT"):
                params["MODEL.BACKBONE.FREEZE_AT"] = cfg.MODEL.BACKBONE.FREEZE_AT
        except (AttributeError, Exception) as e:
            logging.getLogger(__name__).debug(f"Could not log FREEZE_AT: {e}")

        try:
            if hasattr(cfg.MODEL, "ANCHOR_GENERATOR"):
                if hasattr(cfg.MODEL.ANCHOR_GENERATOR, "SIZES"):
                    # Convert nested list to JSON string for MLflow
                    params["MODEL.ANCHOR_GENERATOR.SIZES"] = json.dumps(cfg.MODEL.ANCHOR_GENERATOR.SIZES)
        except (AttributeError, Exception) as e:
            logging.getLogger(__name__).debug(f"Could not log ANCHOR_SIZES: {e}")

        try:
            if hasattr(cfg.MODEL, "ANCHOR_GENERATOR"):
                if hasattr(cfg.MODEL.ANCHOR_GENERATOR, "ASPECT_RATIOS"):
                    # Convert nested list to JSON string for MLflow
                    params["MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"] = json.dumps(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS)
        except (AttributeError, Exception) as e:
            logging.getLogger(__name__).debug(f"Could not log ASPECT_RATIOS: {e}")

        # Log data augmentation parameters
        # These are currently hardcoded in custom_mapper, but we log them for tracking
        try:
            params["AUG.RESIZE_SHORT_EDGE_MIN"] = 640
            params["AUG.RESIZE_SHORT_EDGE_MAX"] = 800
            params["AUG.RESIZE_MAX_SIZE"] = 1333
            params["AUG.RESIZE_SCALES"] = json.dumps([640, 672, 704, 736, 768, 800])
            params["AUG.RANDOM_FLIP_PROB"] = 0.5
            params["AUG.RANDOM_FLIP_HORIZONTAL"] = True
            params["AUG.RANDOM_FLIP_VERTICAL"] = False
            params["AUG.RANDOM_ROTATION_ANGLE_MIN"] = -45
            params["AUG.RANDOM_ROTATION_ANGLE_MAX"] = 45
            params["AUG.RANDOM_BRIGHTNESS_MIN"] = 0.8
            params["AUG.RANDOM_BRIGHTNESS_MAX"] = 1.2
            params["AUG.RANDOM_CONTRAST_MIN"] = 0.8
            params["AUG.RANDOM_CONTRAST_MAX"] = 1.2
        except Exception as e:
            logging.getLogger(__name__).debug(f"Could not log augmentation params: {e}")

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
            MetadataCatalog.get(f"arcade_{split}").set(
                thing_classes=adapter.class_names,
                id_reverse_map={v: k for k, v in adapter.id_map.items()},
                adapter_instance=adapter
            )

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

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"detectron_epochs_{epochs}_batch_{batch}_dt_{timestamp}"):
            trainer = ArcadeTrainer(self.cfg)
            trainer.resume_or_load(resume=False)

            self.log.info(f"Starting training for {epochs} epochs")
            trainer.train()
