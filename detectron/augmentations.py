"""
Detectron2-compatible wrappers for custom augmentations.

This module provides integration between Albumentations-based transforms
and Detectron2's augmentation pipeline, ensuring proper handling of:
- Image transformations
- Polygon/segmentation coordinate transforms
- Bounding box transforms
"""

import numpy as np
from detectron2.data import transforms as T
from typing import Dict, Any, List, Tuple

try:
    from shapely import geometry
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class AddFrameTransform(T.Transform):
    """
    Detectron2-compatible Transform for the AddFrame augmentation.

    This is a deterministic transform that applies pre-sampled parameters
    to images, masks, and coordinates consistently.

    The AddFrame transform either crops or resizes the image content to make
    room for a synthetic frame, then adds the frame around it. The output
    size matches the input size.

    Coordinate handling:
    - Crop method: Coordinates in the visible region are unchanged, but
      polygons must be clipped to exclude the frame area
    - Resize method: Coordinates are scaled by (new_size/orig_size) and
      offset by the frame width
    """

    def __init__(
        self,
        params: Dict[str, Any],
        orig_height: int,
        orig_width: int,
        add_frame_instance,
    ):
        """
        Args:
            params: Pre-sampled parameters from AddFrame.get_params()
            orig_height: Original image height
            orig_width: Original image width
            add_frame_instance: Instance of AddFrame for delegating apply methods
        """
        super().__init__()
        self.params = params
        self.orig_h = orig_height
        self.orig_w = orig_width
        self._add_frame = add_frame_instance

        # Pre-compute frame dimensions for coordinate transforms
        self.top = int(orig_height * params["top_frac"])
        self.bottom = int(orig_height * params["bottom_frac"])
        self.left = int(orig_width * params["left_frac"])
        self.right = int(orig_width * params["right_frac"])
        self.method = params["actual_method"]

        # Size of the content region (after crop/resize, before adding frame)
        self.content_h = orig_height - self.top - self.bottom
        self.content_w = orig_width - self.left - self.right

        # Check for degenerate case
        self.is_valid = self.content_h > 0 and self.content_w > 0

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """Apply the frame transform to an image."""
        if not self.is_valid:
            return img

        # Prepare params with image dimensions (required by Albumentations)
        params = {**self.params, "rows": img.shape[0], "cols": img.shape[1]}
        return self._add_frame.apply(img, **params)

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Apply the transform to a segmentation mask."""
        if not self.is_valid:
            return segmentation

        params = {**self.params, "rows": segmentation.shape[0], "cols": segmentation.shape[1]}
        return self._add_frame.apply_to_mask(segmentation, **params)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform Nx2 coordinate array.

        For crop method: Coordinates are unchanged (frame is added around content
        which stays at the same position). However, coordinates outside the
        visible region should ideally be handled in apply_polygons.

        For resize method: Coordinates are scaled and offset.
        """
        if not self.is_valid or len(coords) == 0:
            return coords

        result = coords.copy().astype(np.float64)

        if self.method == "crop":
            # For crop: content at original [top:H-bottom, left:W-right]
            # ends up at [top:top+content_h, left:left+content_w]
            # The absolute coordinates are preserved for points in the visible region
            # Points outside are in the frame area
            pass  # Coordinates unchanged for crop
        else:
            # For resize: original image is scaled down and placed in center
            scale_x = self.content_w / self.orig_w
            scale_y = self.content_h / self.orig_h
            result[:, 0] = self.left + result[:, 0] * scale_x
            result[:, 1] = self.top + result[:, 1] * scale_y

        return result

    def apply_polygons(self, polygons: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply transform to list of Nx2 polygon coordinate arrays.

        For crop method, polygons must be clipped to the visible region
        (excluding the frame area). This uses Shapely for proper polygon
        intersection which can handle:
        - Partial visibility (polygon extends into frame)
        - Split polygons (when crop divides a polygon)
        - Complete removal (polygon entirely in frame)

        For resize method, just applies coordinate scaling.
        """
        if not self.is_valid:
            return polygons

        if self.method == "crop":
            return self._apply_polygons_crop(polygons)
        else:
            return self._apply_polygons_resize(polygons)

    def _apply_polygons_crop(self, polygons: List[np.ndarray]) -> List[np.ndarray]:
        """Handle polygon clipping for crop method."""
        if not SHAPELY_AVAILABLE:
            # Fallback: just apply coords without clipping
            # This may leave polygon points in the frame region
            return [self.apply_coords(p) for p in polygons if len(p) >= 3]

        # The visible content region in the OUTPUT image
        # Content is placed at [top:top+content_h, left:left+content_w]
        visible_box = geometry.box(
            self.left,  # x_min
            self.top,   # y_min
            self.left + self.content_w,  # x_max
            self.top + self.content_h,   # y_max
        ).buffer(0.0)  # Clean up any numerical issues

        result = []
        for poly_coords in polygons:
            if len(poly_coords) < 3:
                continue

            try:
                # Create Shapely polygon
                poly = geometry.Polygon(poly_coords).buffer(0.0)
                if not poly.is_valid or poly.is_empty:
                    continue

                # Intersect with visible region
                cropped = poly.intersection(visible_box)
                if cropped.is_empty:
                    continue

                # Handle potentially multiple resulting geometries
                if hasattr(cropped, 'geoms'):
                    geoms = list(cropped.geoms)
                else:
                    geoms = [cropped]

                for g in geoms:
                    if isinstance(g, geometry.Polygon) and not g.is_empty:
                        # Get exterior coordinates (remove duplicate closing point)
                        coords = np.array(g.exterior.coords)[:-1]
                        if len(coords) >= 3:
                            # Apply coord transform (no-op for crop, but keep consistent)
                            result.append(self.apply_coords(coords))
                    elif isinstance(g, geometry.MultiPolygon):
                        for sub_poly in g.geoms:
                            if not sub_poly.is_empty:
                                coords = np.array(sub_poly.exterior.coords)[:-1]
                                if len(coords) >= 3:
                                    result.append(self.apply_coords(coords))
            except Exception:
                # If geometry operations fail, skip this polygon
                continue

        return result

    def _apply_polygons_resize(self, polygons: List[np.ndarray]) -> List[np.ndarray]:
        """Handle polygon transformation for resize method."""
        result = []
        for poly_coords in polygons:
            if len(poly_coords) >= 3:
                transformed = self.apply_coords(poly_coords)
                result.append(transformed)
        return result


class AddFrameAugmentation(T.Augmentation):
    """
    Detectron2 Augmentation wrapper for the AddFrame transform.

    This class handles:
    - Probability-based application (respects AddFrame's p parameter)
    - Parameter sampling via AddFrame.get_params()
    - Creation of deterministic AddFrameTransform instances

    Usage in custom_mapper:
        augmentations = [
            T.ResizeShortestEdge(...),
            T.RandomFlip(...),
            AddFrameAugmentation(
                frame_width_range=(0.05, 0.15),
                method="random",
                p=0.5,
            ),
        ]
    """

    def __init__(
        self,
        frame_width_range: Tuple[float, float] = (0.05, 0.15),
        base_gray_range: Tuple[int, int] = (20, 50),
        noise_scale: float = 15.0,
        gradient_strength: float = 0.6,
        gradient_width: float = 0.5,
        use_gradient_effect: bool = False,
        method: str = "crop",
        crop_prob: float = 0.5,
        p: float = 0.5,
    ):
        """
        Args:
            frame_width_range: Range of frame width as fraction of image size
            base_gray_range: Range of base gray value for the frame
            noise_scale: Scale of Gaussian noise added to frame
            gradient_strength: Strength of gradient effect at frame edges
            gradient_width: Width of gradient effect as fraction of frame
            use_gradient_effect: Whether to apply gradient effect
            method: "crop", "resize", or "random"
            crop_prob: Probability of crop when method="random"
            p: Probability of applying this augmentation
        """
        super().__init__()

        # Import here to avoid circular imports
        from frame import AddFrame

        self._add_frame = AddFrame(
            frame_width_range=frame_width_range,
            base_gray_range=base_gray_range,
            noise_scale=noise_scale,
            gradient_strength=gradient_strength,
            gradient_width=gradient_width,
            use_gradient_effect=use_gradient_effect,
            method=method,
            crop_prob=crop_prob,
            p=1.0,  # We handle probability ourselves
        )
        self._prob = p

    def get_transform(self, image: np.ndarray) -> T.Transform:
        """
        Sample parameters and return a deterministic transform.

        Args:
            image: Input image array (H, W, C)

        Returns:
            AddFrameTransform if augmentation is applied, NoOpTransform otherwise
        """
        # Check probability
        if np.random.rand() >= self._prob:
            return T.NoOpTransform()

        # Sample random parameters
        params = self._add_frame.get_params()

        # Get image dimensions
        h, w = image.shape[:2]

        return AddFrameTransform(
            params=params,
            orig_height=h,
            orig_width=w,
            add_frame_instance=self._add_frame,
        )