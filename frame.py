import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform
from typing import Dict, Any, Tuple


class AddFrame(DualTransform):
    def __init__(
        self,
        frame_width_range: Tuple[float, float] = (0.05, 0.15),
        base_gray_range: Tuple[int, int] = (20, 50),
        noise_scale: float = 15.0,
        gradient_strength: float = 0.6,
        gradient_width: float = 0.5,
        use_gradient_effect: bool = False,  # Configurable gradient effect
        method: str = "crop",  # "crop", "resize", or "random"
        crop_prob: float = 0.5,  # Probability of crop when method="random"
        p: float = 0.5,
    ):
        super().__init__(p=p)  # Removed always_apply - deprecated in albumentations
        self.frame_width_range = frame_width_range
        self.base_gray_range = base_gray_range
        self.noise_scale = noise_scale
        self.gradient_strength = gradient_strength
        self.gradient_width = gradient_width
        self.use_gradient_effect = use_gradient_effect
        self.method = method
        self.crop_prob = crop_prob

        if method not in ["crop", "resize", "random"]:
            raise ValueError(
                f"method must be 'crop', 'resize', or 'random', got '{method}'"
            )

    def get_params(self) -> Dict[str, Any]:
        """Generate random parameters for this transform."""
        frame_width_frac = np.random.uniform(*self.frame_width_range)

        top_frac = frame_width_frac * np.random.uniform(0.5, 1.5)
        bottom_frac = frame_width_frac * np.random.uniform(0.5, 1.5)
        left_frac = frame_width_frac * np.random.uniform(0.5, 1.5)
        right_frac = frame_width_frac * np.random.uniform(0.5, 1.5)

        base_gray = int(np.random.uniform(*self.base_gray_range))

        actual_method = self.method
        if self.method == "random":
            actual_method = "crop" if np.random.rand() < self.crop_prob else "resize"

        return {
            "top_frac": top_frac,
            "bottom_frac": bottom_frac,
            "left_frac": left_frac,
            "right_frac": right_frac,
            "base_gray": base_gray,
            "actual_method": actual_method,  # Store the chosen method
        }

    @property
    def targets_as_params(self):
        """Return list of targets that are used as parameters."""
        return []

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate pixel-based frame widths from fractions based on image size."""
        return {}

    @property
    def targets(self):
        """Specify transform targets (image, mask, and bounding boxes)."""
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "bboxes": self.apply_to_bboxes,
        }

    def apply_to_bboxes(self, bboxes, **params):
        """Apply transform to a list/array of bounding boxes."""
        transformed = [self.apply_to_bbox(bbox, **params) for bbox in bboxes]
        return np.asarray(transformed, dtype=np.float32)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Apply angiographic frame to image.

        Three methods available:
        - "crop": Crop image edges, add frame (preserves 100% sharpness of center)
        - "resize": Shrink image, add frame (may reduce sharpness, but uses aggressive sharpening)
        - "random": Randomly choose between crop and resize (controlled by crop_prob)

        All maintain constant image size (important for training).
        """
        top_frac = params["top_frac"]
        bottom_frac = params["bottom_frac"]
        left_frac = params["left_frac"]
        right_frac = params["right_frac"]
        base_gray = params["base_gray"]
        actual_method = params["actual_method"]

        orig_h, orig_w = img.shape[:2]

        frame_widths = {
            "top": int(orig_h * top_frac),
            "bottom": int(orig_h * bottom_frac),
            "left": int(orig_w * left_frac),
            "right": int(orig_w * right_frac),
        }

        new_h = orig_h - frame_widths["top"] - frame_widths["bottom"]
        new_w = orig_w - frame_widths["left"] - frame_widths["right"]

        if new_h <= 0 or new_w <= 0:
            return img

        if actual_method == "crop":
            # CROP METHOD: Cut edges (preserves 100% sharpness of center)
            # Calculate crop coordinates (from center)
            crop_top = frame_widths["top"]
            crop_bottom = orig_h - frame_widths["bottom"]
            crop_left = frame_widths["left"]
            crop_right = orig_w - frame_widths["right"]

            # Crop center region
            center_img = img[crop_top:crop_bottom, crop_left:crop_right]

        else:
            center_img = cv2.resize(
                img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )

            if new_h < orig_h * 0.95:
                blurred_fine = cv2.GaussianBlur(center_img, (3, 3), 0.5)
                center_img = cv2.addWeighted(center_img, 2.2, blurred_fine, -1.2, 0)
                center_img = np.clip(center_img, 0, 255).astype(img.dtype)

                blurred_coarse = cv2.GaussianBlur(center_img, (5, 5), 1.0)
                center_img = cv2.addWeighted(center_img, 1.3, blurred_coarse, -0.3, 0)
                center_img = np.clip(center_img, 0, 255).astype(img.dtype)

        return self._add_frame(center_img, frame_widths, base_gray)

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        """
        Apply same transformation to mask (background class = 0).

        Three methods:
        - "crop": Crop mask edges, add padding (background=0)
        - "resize": Shrink mask, add padding (background=0)
        - "random": Randomly choose between crop and resize (same as image)

        Handles both 2D (H, W) and 3D (H, W, 1) masks.
        """
        top_frac = params["top_frac"]
        bottom_frac = params["bottom_frac"]
        left_frac = params["left_frac"]
        right_frac = params["right_frac"]
        actual_method = params["actual_method"]

        is_3d = mask.ndim == 3
        if is_3d:
            mask_2d = mask.squeeze(axis=2) if mask.shape[2] == 1 else mask[..., 0]
        else:
            mask_2d = mask

        orig_h, orig_w = mask_2d.shape[:2]

        top = int(orig_h * top_frac)
        bottom = int(orig_h * bottom_frac)
        left = int(orig_w * left_frac)
        right = int(orig_w * right_frac)

        new_h = orig_h - top - bottom
        new_w = orig_w - left - right

        if new_h <= 0 or new_w <= 0:
            return mask

        if actual_method == "crop":
            crop_top = top
            crop_bottom = orig_h - bottom
            crop_left = left
            crop_right = orig_w - right
            center_mask = mask_2d[crop_top:crop_bottom, crop_left:crop_right]

        else:
            center_mask = cv2.resize(
                mask_2d, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )

        result = cv2.copyMakeBorder(
            center_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

        if is_3d:
            result = result[..., np.newaxis]

        return result

    def apply_to_bbox(self, bbox: Tuple[float, float, float, float], **params):
        """Transform bounding box consistently with image/mask."""
        if bbox is None:
            return bbox

        extra = None

        if isinstance(bbox, np.ndarray):
            bbox_values = bbox.astype(np.float32)
            if bbox_values.size < 4:
                return bbox
            x_min, y_min, x_max, y_max = bbox_values[:4].tolist()
            extra = bbox_values[4:]
        else:
            if len(bbox) < 4:
                return bbox
            x_min, y_min, x_max, y_max = [float(v) for v in bbox[:4]]
            extra = bbox[4:]

        top_frac = params["top_frac"]
        bottom_frac = params["bottom_frac"]
        left_frac = params["left_frac"]
        right_frac = params["right_frac"]
        actual_method = params["actual_method"]

        orig_h = params.get("rows")
        orig_w = params.get("cols")

        if orig_h is None or orig_w is None:
            return bbox

        # If albumentations passes normalized coords (rare with pascal_voc), detect and convert.
        is_normalized_input = max(x_max, y_max) <= 1.0 + 1e-6
        if is_normalized_input:
            x_min *= orig_w
            y_min *= orig_h
            x_max *= orig_w
            y_max *= orig_h

        top = int(orig_h * top_frac)
        bottom = int(orig_h * bottom_frac)
        left = int(orig_w * left_frac)
        right = int(orig_w * right_frac)

        new_h = orig_h - top - bottom
        new_w = orig_w - left - right

        if new_h <= 0 or new_w <= 0:
            return bbox

        x_min_new, y_min_new, x_max_new, y_max_new = x_min, y_min, x_max, y_max

        if actual_method == "crop":
            x_min_new = np.clip(x_min_new, left, orig_w - right)
            x_max_new = np.clip(x_max_new, left, orig_w - right)
            y_min_new = np.clip(y_min_new, top, orig_h - bottom)
            y_max_new = np.clip(y_max_new, top, orig_h - bottom)
        else:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

            x_min_new = left + x_min_new * scale_x
            x_max_new = left + x_max_new * scale_x
            y_min_new = top + y_min_new * scale_y
            y_max_new = top + y_max_new * scale_y

        eps = 1e-3

        max_x = orig_w - eps
        max_y = orig_h - eps

        x_min_new = float(np.clip(x_min_new, 0.0, max_x))
        x_max_new = float(np.clip(x_max_new, x_min_new + eps, max_x))
        y_min_new = float(np.clip(y_min_new, 0.0, max_y))
        y_max_new = float(np.clip(y_max_new, y_min_new + eps, max_y))

        if x_max_new - x_min_new < eps:
            centre_x = np.clip((x_min_new + x_max_new) / 2.0, 0.0, max_x)
            x_min_new = float(np.clip(centre_x - eps, 0.0, max_x))
            x_max_new = float(np.clip(centre_x + eps, x_min_new + eps, max_x))

        if y_max_new - y_min_new < eps:
            centre_y = np.clip((y_min_new + y_max_new) / 2.0, 0.0, max_y)
            y_min_new = float(np.clip(centre_y - eps, 0.0, max_y))
            y_max_new = float(np.clip(centre_y + eps, y_min_new + eps, max_y))

        if is_normalized_input:
            transformed = (
                x_min_new / orig_w,
                y_min_new / orig_h,
                x_max_new / orig_w,
                y_max_new / orig_h,
            )
        else:
            transformed = (x_min_new, y_min_new, x_max_new, y_max_new)

        if extra is not None and len(extra) > 0:
            if isinstance(extra, np.ndarray):
                extra = extra.tolist()
            return tuple(list(transformed) + list(extra))

        return transformed

    def _add_frame(
        self, img: np.ndarray, frame_widths: Dict[str, int], base_gray: int
    ) -> np.ndarray:
        """
        Add realistic angiographic frame with lighting variation and noise.

        The frame consists of:
        1. Base gray value
        2. Non-uniform brightness (smooth, organic lighting variation using large-scale blur)
        3. Medium blur for smooth appearance
        4. Gaussian noise (adds realistic texture)

        This creates a realistic medical imaging frame with natural-looking uneven illumination.
        """
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        top, bottom, left, right = (
            frame_widths["top"],
            frame_widths["bottom"],
            frame_widths["left"],
            frame_widths["right"],
        )

        new_h = h + top + bottom
        new_w = w + left + right

        if channels == 1:
            framed_img = np.full((new_h, new_w), base_gray, dtype=img.dtype)
        else:
            framed_img = np.full((new_h, new_w, channels), base_gray, dtype=img.dtype)

        h_frame, w_frame = framed_img.shape[:2]

        brightness_variation = np.random.randn(h_frame, w_frame).astype(np.float32)
        brightness_variation = cv2.GaussianBlur(brightness_variation, (51, 51), 25)
        brightness_variation = brightness_variation / np.std(brightness_variation) * 4.0

        if channels > 1:
            brightness_variation = np.stack([brightness_variation] * channels, axis=-1)
        framed_img = np.clip(
            framed_img.astype(np.float32) + brightness_variation, 0, 255
        ).astype(img.dtype)

        kernel_size = 11
        framed_img = cv2.GaussianBlur(framed_img, (kernel_size, kernel_size), 0)

        if self.noise_scale > 0:
            noise = np.random.normal(0, self.noise_scale, framed_img.shape).astype(
                np.float32
            )
            framed_img = np.clip(framed_img.astype(np.float32) + noise, 0, 255).astype(
                img.dtype
            )

        # Place original image in center
        framed_img[top : top + h, left : left + w] = img

        # Apply gradient effect if enabled
        if self.use_gradient_effect:
            framed_img = self._add_gradient_effect(
                framed_img, img, top, bottom, left, right, h, w
            )

        return framed_img

    def _add_gradient_effect(
        self,
        framed_img: np.ndarray,
        original_img: np.ndarray,
        top: int,
        bottom: int,
        left: int,
        right: int,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """
        Add gradient effect where light from image edges "bleeds" into the frame.

        This creates a more realistic transition between the image and frame,
        simulating how X-ray equipment produces gradual intensity falloff at edges.
        Includes corner gradients for smooth 2D transitions.
        """
        edge_top = original_img[0, :].mean() if top > 0 else 0
        edge_bottom = original_img[-1, :].mean() if bottom > 0 else 0
        edge_left = original_img[:, 0].mean() if left > 0 else 0
        edge_right = original_img[:, -1].mean() if right > 0 else 0

        corner_tl = (edge_top + edge_left) / 2  # Top-left
        corner_tr = (edge_top + edge_right) / 2  # Top-right
        corner_bl = (edge_bottom + edge_left) / 2  # Bottom-left
        corner_br = (edge_bottom + edge_right) / 2  # Bottom-right

        if top > 0:
            gradient_height = int(top * self.gradient_width)
            if gradient_height > 0:
                gradient = (
                    np.linspace(1, 0, gradient_height) ** 1.5
                )  # Non-linear falloff
                for i in range(gradient_height):
                    blend_strength = (
                        gradient[gradient_height - 1 - i] * self.gradient_strength
                    )
                    framed_img[i, left : left + orig_w] = (
                        blend_strength * edge_top
                        + (1 - blend_strength) * framed_img[i, left : left + orig_w]
                    )

        if bottom > 0:
            gradient_height = int(bottom * self.gradient_width)
            if gradient_height > 0:
                gradient = np.linspace(1, 0, gradient_height) ** 1.5
                for i in range(gradient_height):
                    row_idx = top + orig_h + i
                    blend_strength = gradient[i] * self.gradient_strength
                    framed_img[row_idx, left : left + orig_w] = (
                        blend_strength * edge_bottom
                        + (1 - blend_strength)
                        * framed_img[row_idx, left : left + orig_w]
                    )

        if left > 0:
            gradient_width = int(left * self.gradient_width)
            if gradient_width > 0:
                gradient = np.linspace(1, 0, gradient_width) ** 1.5
                for i in range(gradient_width):
                    blend_strength = (
                        gradient[gradient_width - 1 - i] * self.gradient_strength
                    )
                    framed_img[top : top + orig_h, i] = (
                        blend_strength * edge_left
                        + (1 - blend_strength) * framed_img[top : top + orig_h, i]
                    )

        if right > 0:
            gradient_width = int(right * self.gradient_width)
            if gradient_width > 0:
                gradient = np.linspace(1, 0, gradient_width) ** 1.5
                for i in range(gradient_width):
                    col_idx = left + orig_w + i
                    blend_strength = gradient[i] * self.gradient_strength
                    framed_img[top : top + orig_h, col_idx] = (
                        blend_strength * edge_right
                        + (1 - blend_strength) * framed_img[top : top + orig_h, col_idx]
                    )

        # Vectorized corner gradients (replaces 4 nested loop sections)
        # Top-left corner
        if top > 0 and left > 0:
            grad_h = int(top * self.gradient_width)
            grad_w = int(left * self.gradient_width)
            if grad_h > 0 and grad_w > 0:
                i_coords, j_coords = np.meshgrid(
                    range(grad_h), range(grad_w), indexing="ij"
                )
                dist_v = (grad_h - i_coords) / grad_h
                dist_h = (grad_w - j_coords) / grad_w
                dist_combined = np.sqrt(dist_v**2 + dist_h**2) / np.sqrt(2)
                dist_combined = np.minimum(1.0, dist_combined)
                blend_strength = (dist_combined**1.5) * self.gradient_strength

                current_values = framed_img[:grad_h, :grad_w]
                # Handle multi-channel broadcasting
                if len(current_values.shape) == 3:
                    blend_strength = blend_strength[..., np.newaxis]
                framed_img[:grad_h, :grad_w] = (
                    blend_strength * corner_tl + (1 - blend_strength) * current_values
                )

        # Top-right corner
        if top > 0 and right > 0:
            grad_h = int(top * self.gradient_width)
            grad_w = int(right * self.gradient_width)
            if grad_h > 0 and grad_w > 0:
                i_coords, j_coords = np.meshgrid(
                    range(grad_h), range(grad_w), indexing="ij"
                )
                dist_v = (grad_h - i_coords) / grad_h
                dist_h = (j_coords + 1) / grad_w
                dist_combined = np.sqrt(dist_v**2 + dist_h**2) / np.sqrt(2)
                dist_combined = np.minimum(1.0, dist_combined)
                blend_strength = (dist_combined**1.5) * self.gradient_strength

                col_start = left + orig_w
                current_values = framed_img[:grad_h, col_start : col_start + grad_w]
                # Handle multi-channel broadcasting
                if len(current_values.shape) == 3:
                    blend_strength = blend_strength[..., np.newaxis]
                framed_img[:grad_h, col_start : col_start + grad_w] = (
                    blend_strength * corner_tr + (1 - blend_strength) * current_values
                )

        # Bottom-left corner
        if bottom > 0 and left > 0:
            grad_h = int(bottom * self.gradient_width)
            grad_w = int(left * self.gradient_width)
            if grad_h > 0 and grad_w > 0:
                i_coords, j_coords = np.meshgrid(
                    range(grad_h), range(grad_w), indexing="ij"
                )
                dist_v = (i_coords + 1) / grad_h
                dist_h = (grad_w - j_coords) / grad_w
                dist_combined = np.sqrt(dist_v**2 + dist_h**2) / np.sqrt(2)
                dist_combined = np.minimum(1.0, dist_combined)
                blend_strength = (dist_combined**1.5) * self.gradient_strength

                row_start = top + orig_h
                current_values = framed_img[row_start : row_start + grad_h, :grad_w]
                # Handle multi-channel broadcasting
                if len(current_values.shape) == 3:
                    blend_strength = blend_strength[..., np.newaxis]
                framed_img[row_start : row_start + grad_h, :grad_w] = (
                    blend_strength * corner_bl + (1 - blend_strength) * current_values
                )

        # Bottom-right corner
        if bottom > 0 and right > 0:
            grad_h = int(bottom * self.gradient_width)
            grad_w = int(right * self.gradient_width)
            if grad_h > 0 and grad_w > 0:
                i_coords, j_coords = np.meshgrid(
                    range(grad_h), range(grad_w), indexing="ij"
                )
                dist_v = (i_coords + 1) / grad_h
                dist_h = (j_coords + 1) / grad_w
                dist_combined = np.sqrt(dist_v**2 + dist_h**2) / np.sqrt(2)
                dist_combined = np.minimum(1.0, dist_combined)
                blend_strength = (dist_combined**1.5) * self.gradient_strength

                row_start = top + orig_h
                col_start = left + orig_w
                current_values = framed_img[
                    row_start : row_start + grad_h, col_start : col_start + grad_w
                ]
                # Handle multi-channel broadcasting
                if len(current_values.shape) == 3:
                    blend_strength = blend_strength[..., np.newaxis]
                framed_img[
                    row_start : row_start + grad_h, col_start : col_start + grad_w
                ] = (blend_strength * corner_br + (1 - blend_strength) * current_values)

        return framed_img

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Return names of arguments used in __init__."""
        return (
            "frame_width_range",
            "base_gray_range",
            "noise_scale",
            "gradient_strength",
            "gradient_width",
            "use_gradient_effect",
            "method",
            "crop_prob",
        )


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    print("Testing frame augmentation...")
    path = input("Enter path to test image: ").strip()

    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    print(f"[INFO] Loading image: {path.name}")
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    original_shape = image.shape[:2]
    print(f"[INFO] Original shape: {original_shape}")

    if image.shape[:2] != (1024, 1024):
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        print(f"[INFO] Resized to: 1024x1024")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    AUGMENTATION_PARAMS = {
        "HORIZONTAL_FLIP_P": 0.5,
        "VERTICAL_FLIP_P": 0.5,
        "RANDOM_ROTATE_90_P": 0.5,
        "ROTATE_LIMIT": 15,
        "ROTATE_P": 0.3,
        "ELASTIC_TRANSFORM_ALPHA": 1.0,
        "ELASTIC_TRANSFORM_SIGMA": 50.0,
        "ELASTIC_TRANSFORM_P": 0.2,
        "GRID_DISTORTION_P": 0.2,
        "OPTICAL_DISTORTION_LIMIT": 0,
        "OPTICAL_DISTORTION_P": 0.2,
        "RANDOM_BRC_B_LIMIT": 0.1,
        "RANDOM_BRC_C_LIMIT": 0.1,
        "RANDOM_BRC_P": 0.3,
        "GAUSS_NOISE_P": 0.2,
        "BLUR_LIMIT": 5,
        "BLUR_P": 0.2,
    }

    augment = A.Compose(
        [
            A.HorizontalFlip(p=AUGMENTATION_PARAMS["HORIZONTAL_FLIP_P"]),
            A.VerticalFlip(p=AUGMENTATION_PARAMS["VERTICAL_FLIP_P"]),
            A.RandomRotate90(p=AUGMENTATION_PARAMS["RANDOM_ROTATE_90_P"]),
            A.Rotate(
                limit=AUGMENTATION_PARAMS["ROTATE_LIMIT"],
                p=AUGMENTATION_PARAMS["ROTATE_P"],
            ),
            A.ElasticTransform(
                alpha=AUGMENTATION_PARAMS["ELASTIC_TRANSFORM_ALPHA"],
                sigma=AUGMENTATION_PARAMS["ELASTIC_TRANSFORM_SIGMA"],
                p=AUGMENTATION_PARAMS["ELASTIC_TRANSFORM_P"],
            ),
            A.GridDistortion(p=AUGMENTATION_PARAMS["GRID_DISTORTION_P"]),
            A.OpticalDistortion(
                distort_limit=AUGMENTATION_PARAMS["OPTICAL_DISTORTION_LIMIT"],
                p=AUGMENTATION_PARAMS["OPTICAL_DISTORTION_P"],
            ),
            A.RandomBrightnessContrast(
                brightness_limit=AUGMENTATION_PARAMS["RANDOM_BRC_B_LIMIT"],
                contrast_limit=AUGMENTATION_PARAMS["RANDOM_BRC_C_LIMIT"],
                p=AUGMENTATION_PARAMS["RANDOM_BRC_P"],
            ),
            A.GaussNoise(p=AUGMENTATION_PARAMS["GAUSS_NOISE_P"]),
            A.Blur(
                blur_limit=AUGMENTATION_PARAMS["BLUR_LIMIT"],
                p=AUGMENTATION_PARAMS["BLUR_P"],
            ),
            AddFrame(
                frame_width_range=(0.05, 0.15),
                base_gray_range=(20, 50),
                noise_scale=15,
                gradient_strength=0.6,
                gradient_width=0.5,
                p=0.5,
            ),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )

    augmented = augment(image=image_rgb)
    augmented_image = augmented["image"].numpy().transpose(1, 2, 0)
    augmented_image = ((augmented_image * 0.5) + 0.5) * 255.0
    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Original Image", image)
    cv2.imshow("Augmented Image with Frame", augmented_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
