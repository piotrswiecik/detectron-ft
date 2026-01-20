"""
Visual test for AddFrame augmentation.

This script demonstrates:
1. Image transformation (with frame added)
2. Polygon/annotation transformation
3. Side-by-side comparison of original vs augmented

Usage:
    python test_augmentation_visual.py [image_path]

If no image path provided, generates a synthetic test image.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

from frame import AddFrame

# Standalone polygon transform logic (mirrors detectron/augmentations.py)
# This avoids requiring detectron2 for visual testing

try:
    from shapely import geometry
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not installed. Polygon clipping will be approximate.")


class StandaloneAddFrameTransform:
    """Standalone transform for testing without detectron2."""

    def __init__(self, params, orig_height, orig_width, add_frame_instance):
        self.params = params
        self.orig_h = orig_height
        self.orig_w = orig_width
        self._add_frame = add_frame_instance

        self.top = int(orig_height * params["top_frac"])
        self.bottom = int(orig_height * params["bottom_frac"])
        self.left = int(orig_width * params["left_frac"])
        self.right = int(orig_width * params["right_frac"])
        self.method = params["actual_method"]

        self.content_h = orig_height - self.top - self.bottom
        self.content_w = orig_width - self.left - self.right
        self.is_valid = self.content_h > 0 and self.content_w > 0

    def apply_image(self, img):
        if not self.is_valid:
            return img
        params = {**self.params, "rows": img.shape[0], "cols": img.shape[1]}
        return self._add_frame.apply(img, **params)

    def apply_coords(self, coords):
        if not self.is_valid or len(coords) == 0:
            return coords
        result = coords.copy().astype(np.float64)
        if self.method == "crop":
            pass  # Coords unchanged for crop
        else:
            scale_x = self.content_w / self.orig_w
            scale_y = self.content_h / self.orig_h
            result[:, 0] = self.left + result[:, 0] * scale_x
            result[:, 1] = self.top + result[:, 1] * scale_y
        return result

    def apply_polygons(self, polygons):
        if not self.is_valid:
            return polygons
        if self.method == "crop":
            return self._apply_polygons_crop(polygons)
        else:
            return self._apply_polygons_resize(polygons)

    def _apply_polygons_crop(self, polygons):
        if not SHAPELY_AVAILABLE:
            return [self.apply_coords(p) for p in polygons if len(p) >= 3]

        visible_box = geometry.box(
            self.left, self.top,
            self.left + self.content_w, self.top + self.content_h,
        ).buffer(0.0)

        result = []
        for poly_coords in polygons:
            if len(poly_coords) < 3:
                continue
            try:
                poly = geometry.Polygon(poly_coords).buffer(0.0)
                if not poly.is_valid or poly.is_empty:
                    continue
                cropped = poly.intersection(visible_box)
                if cropped.is_empty:
                    continue
                if hasattr(cropped, 'geoms'):
                    geoms = list(cropped.geoms)
                else:
                    geoms = [cropped]
                for g in geoms:
                    if isinstance(g, geometry.Polygon) and not g.is_empty:
                        coords = np.array(g.exterior.coords)[:-1]
                        if len(coords) >= 3:
                            result.append(self.apply_coords(coords))
            except Exception:
                continue
        return result

    def _apply_polygons_resize(self, polygons):
        return [self.apply_coords(p) for p in polygons if len(p) >= 3]


# Use standalone transform for testing
AddFrameTransform = StandaloneAddFrameTransform


def create_synthetic_image(height=512, width=768):
    """Create a synthetic angiography-like image for testing."""
    # Dark background
    img = np.full((height, width, 3), 30, dtype=np.uint8)

    # Add some vessel-like structures (bright curved lines)
    for _ in range(5):
        # Random bezier-like curve
        start = (np.random.randint(50, width-50), np.random.randint(50, height-50))
        end = (np.random.randint(50, width-50), np.random.randint(50, height-50))
        ctrl = (np.random.randint(50, width-50), np.random.randint(50, height-50))

        # Draw curve as series of points
        pts = []
        for t in np.linspace(0, 1, 50):
            x = int((1-t)**2 * start[0] + 2*(1-t)*t * ctrl[0] + t**2 * end[0])
            y = int((1-t)**2 * start[1] + 2*(1-t)*t * ctrl[1] + t**2 * end[1])
            pts.append((x, y))

        # Draw the curve
        brightness = np.random.randint(150, 255)
        thickness = np.random.randint(2, 6)
        for i in range(len(pts)-1):
            cv2.line(img, pts[i], pts[i+1], (brightness, brightness, brightness), thickness)

    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def create_test_polygons(height, width):
    """Create test polygon annotations."""
    polygons = []

    # Polygon 1: Rectangle in center
    margin = 0.2
    p1 = np.array([
        [width * margin, height * margin],
        [width * (1-margin), height * margin],
        [width * (1-margin), height * (1-margin)],
        [width * margin, height * (1-margin)],
    ])
    polygons.append(p1)

    # Polygon 2: Triangle in top-left (might be affected by frame)
    p2 = np.array([
        [50, 50],
        [150, 50],
        [100, 150],
    ])
    polygons.append(p2)

    # Polygon 3: Small square near bottom-right edge
    p3 = np.array([
        [width - 100, height - 100],
        [width - 30, height - 100],
        [width - 30, height - 30],
        [width - 100, height - 30],
    ])
    polygons.append(p3)

    # Polygon 4: Elongated shape crossing left edge
    p4 = np.array([
        [20, height//2 - 50],
        [200, height//2 - 30],
        [200, height//2 + 30],
        [20, height//2 + 50],
    ])
    polygons.append(p4)

    return polygons


def draw_polygons(img, polygons, colors=None):
    """Draw polygons on image."""
    img_copy = img.copy()
    if colors is None:
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]

    for i, poly in enumerate(polygons):
        color = colors[i % len(colors)]
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts], True, color, 2)
        # Fill with transparency
        overlay = img_copy.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, img_copy, 0.7, 0, img_copy)

    return img_copy


def draw_frame_boundary(img, transform):
    """Draw the frame boundary (visible region) on image."""
    img_copy = img.copy()
    h, w = img.shape[:2]

    # Draw frame boundary as dashed rectangle
    left = transform.left
    top = transform.top
    right = w - transform.right
    bottom = h - transform.bottom

    # Draw dashed rectangle
    color = (0, 255, 255)  # Yellow
    thickness = 2

    # Top line
    for x in range(left, right, 20):
        cv2.line(img_copy, (x, top), (min(x+10, right), top), color, thickness)
    # Bottom line
    for x in range(left, right, 20):
        cv2.line(img_copy, (x, bottom), (min(x+10, right), bottom), color, thickness)
    # Left line
    for y in range(top, bottom, 20):
        cv2.line(img_copy, (left, y), (left, min(y+10, bottom)), color, thickness)
    # Right line
    for y in range(top, bottom, 20):
        cv2.line(img_copy, (right, y), (right, min(y+10, bottom)), color, thickness)

    return img_copy


def visualize_augmentation(image_path=None, method="crop"):
    """Main visualization function."""
    print(f"Testing AddFrame augmentation with method='{method}'")
    print("=" * 60)

    # Load or create image
    if image_path and Path(image_path).exists():
        img = cv2.imread(str(image_path))
        print(f"Loaded image: {image_path}")
    else:
        print("Creating synthetic test image...")
        img = create_synthetic_image(512, 768)

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Create test polygons
    polygons = create_test_polygons(h, w)
    print(f"Created {len(polygons)} test polygons")

    # Create augmentation with fixed parameters for reproducibility
    add_frame = AddFrame(p=1.0, method=method)
    params = {
        'top_frac': 0.1,
        'bottom_frac': 0.1,
        'left_frac': 0.1,
        'right_frac': 0.1,
        'base_gray': 30,
        'actual_method': method,
    }

    transform = AddFrameTransform(params, h, w, add_frame)

    print(f"\nFrame parameters:")
    print(f"  Top: {transform.top}px ({params['top_frac']*100:.0f}%)")
    print(f"  Bottom: {transform.bottom}px ({params['bottom_frac']*100:.0f}%)")
    print(f"  Left: {transform.left}px ({params['left_frac']*100:.0f}%)")
    print(f"  Right: {transform.right}px ({params['right_frac']*100:.0f}%)")
    print(f"  Content region: {transform.content_w}x{transform.content_h}")

    # Apply transforms
    img_transformed = transform.apply_image(img)
    polygons_transformed = transform.apply_polygons(polygons)

    print(f"\nPolygon transformation:")
    print(f"  Input: {len(polygons)} polygons")
    print(f"  Output: {len(polygons_transformed)} polygons")

    # Draw visualizations
    img_orig_with_poly = draw_polygons(img, polygons)
    img_orig_with_boundary = draw_frame_boundary(img_orig_with_poly, transform)

    img_trans_with_poly = draw_polygons(img_transformed, polygons_transformed)

    # Create side-by-side comparison
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_orig_with_boundary, f"Original (method={method})", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img_orig_with_boundary, "Yellow dashed = crop boundary", (10, 55), font, 0.5, (0, 255, 255), 1)
    cv2.putText(img_trans_with_poly, "Transformed", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img_trans_with_poly, f"Polygons: {len(polygons)} -> {len(polygons_transformed)}", (10, 55), font, 0.5, (255, 255, 255), 1)

    # Stack horizontally
    comparison = np.hstack([img_orig_with_boundary, img_trans_with_poly])

    # Show
    cv2.imshow(f"AddFrame Augmentation Test (method={method})", comparison)
    print("\nPress any key to continue, 'q' to quit, 's' to save...")

    key = cv2.waitKey(0) & 0xFF

    if key == ord('s'):
        output_path = f"augmentation_test_{method}.png"
        cv2.imwrite(output_path, comparison)
        print(f"Saved to {output_path}")

    return key != ord('q')


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n" + "="*60)
    print("AddFrame Augmentation Visual Test")
    print("="*60)
    print("\nThis test shows how the augmentation affects:")
    print("  - Image pixels (frame added around content)")
    print("  - Polygon annotations (clipped/transformed)")
    print("\nPolygon colors:")
    print("  - Green: Center rectangle (should be fully visible)")
    print("  - Blue: Top-left triangle (may be clipped)")
    print("  - Red: Bottom-right square (may be clipped)")
    print("  - Cyan: Left-edge shape (should be clipped)")
    print()

    # Test crop method
    if not visualize_augmentation(image_path, method="crop"):
        cv2.destroyAllWindows()
        return

    # Test resize method
    if not visualize_augmentation(image_path, method="resize"):
        cv2.destroyAllWindows()
        return

    cv2.destroyAllWindows()
    print("\nTest complete!")


if __name__ == "__main__":
    main()