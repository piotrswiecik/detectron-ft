"""
Unit tests for converters.py

Run with: pytest test_converters.py -v
"""

import numpy as np
import pytest
from converters import DetectronToArcadeConverter, ArcadeMetricsCalculator


class TestSegmentationNormalization:
    """Test segmentation format handling."""

    def test_flat_segmentation(self):
        """Flat list should pass through unchanged."""
        calc = ArcadeMetricsCalculator()
        flat = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0]
        result = calc.normalize_segmentation(flat)
        assert result == flat

    def test_nested_segmentation(self):
        """Nested list should be unwrapped."""
        calc = ArcadeMetricsCalculator()
        nested = [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0]]
        result = calc.normalize_segmentation(nested)
        assert result == [10.0, 10.0, 50.0, 10.0, 50.0, 50.0]

    def test_empty_segmentation(self):
        """Empty list should return empty."""
        calc = ArcadeMetricsCalculator()
        assert calc.normalize_segmentation([]) == []


class TestPolygonToMask:
    """Test polygon to mask conversion."""

    def test_simple_square(self):
        """Simple square polygon."""
        calc = ArcadeMetricsCalculator()
        segmentation = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]
        mask = calc.polygon_to_mask(segmentation, height=100, width=100)

        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        assert mask[30, 30] == True  # Interior
        assert mask[5, 5] == False   # Exterior

    def test_nested_segmentation_input(self):
        """Should handle nested segmentation format."""
        calc = ArcadeMetricsCalculator()
        segmentation = [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]]
        mask = calc.polygon_to_mask(segmentation, height=100, width=100)

        assert mask.shape == (100, 100)
        assert mask[30, 30] == True

    def test_insufficient_points(self):
        """Less than 3 points should return empty mask."""
        calc = ArcadeMetricsCalculator()
        segmentation = [10.0, 10.0, 20.0, 20.0]  # Only 2 points
        mask = calc.polygon_to_mask(segmentation, height=100, width=100)

        assert mask.sum() == 0

    def test_empty_segmentation(self):
        """Empty segmentation returns empty mask."""
        calc = ArcadeMetricsCalculator()
        mask = calc.polygon_to_mask([], height=100, width=100)
        assert mask.sum() == 0


class TestMaskToPolygon:
    """Test mask to polygon conversion."""

    def test_simple_mask(self):
        """Simple square mask."""
        converter = DetectronToArcadeConverter({0: 0})
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        polygon = converter._mask_to_polygon(mask)

        assert len(polygon) >= 8  # At least 4 corners
        coords = np.array(polygon).reshape(-1, 2)
        assert coords[:, 0].min() >= 0
        assert coords[:, 0].max() < 100

    def test_float_mask(self):
        """Float mask with values 0-1."""
        converter = DetectronToArcadeConverter({0: 0})
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 0.9

        polygon = converter._mask_to_polygon(mask)
        assert len(polygon) >= 8

    def test_empty_mask(self):
        """Empty mask returns empty polygon."""
        converter = DetectronToArcadeConverter({0: 0})
        mask = np.zeros((100, 100), dtype=bool)

        polygon = converter._mask_to_polygon(mask)
        assert polygon == []


class TestScalePolygon:
    """Test polygon coordinate scaling."""

    def test_scale_up(self):
        """Scale up by 2x."""
        polygon = [10.0, 20.0, 30.0, 40.0]
        scaled = DetectronToArcadeConverter._scale_polygon(polygon, 2.0, 2.0)
        assert scaled == [20.0, 40.0, 60.0, 80.0]

    def test_scale_down(self):
        """Scale down by 0.5x."""
        polygon = [100.0, 200.0, 300.0, 400.0]
        scaled = DetectronToArcadeConverter._scale_polygon(polygon, 0.5, 0.5)
        assert scaled == [50.0, 100.0, 150.0, 200.0]

    def test_asymmetric_scale(self):
        """Different X and Y scale factors."""
        polygon = [100.0, 100.0]
        scaled = DetectronToArcadeConverter._scale_polygon(polygon, 0.5, 2.0)
        assert scaled == [50.0, 200.0]


class TestIoUCalculation:
    """Test IoU calculation."""

    def test_perfect_overlap(self):
        calc = ArcadeMetricsCalculator()
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        iou = calc.calculate_iou(mask, mask.copy())
        assert iou == 1.0

    def test_no_overlap(self):
        calc = ArcadeMetricsCalculator()
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[10:20, 10:20] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[80:90, 80:90] = True

        iou = calc.calculate_iou(mask1, mask2)
        assert iou == 0.0

    def test_partial_overlap(self):
        calc = ArcadeMetricsCalculator()
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[0:50, 0:100] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[25:75, 0:100] = True

        iou = calc.calculate_iou(mask1, mask2)
        # Intersection: 25 rows, Union: 75 rows
        assert 0.3 < iou < 0.4


class TestDiceCalculation:
    """Test Dice coefficient calculation."""

    def test_perfect_overlap(self):
        calc = ArcadeMetricsCalculator()
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        dice = calc.calculate_dice(mask, mask.copy())
        assert dice == 1.0

    def test_no_overlap(self):
        calc = ArcadeMetricsCalculator()
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[10:20, 10:20] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[80:90, 80:90] = True

        dice = calc.calculate_dice(mask1, mask2)
        assert dice == 0.0


class TestHungarianMatching:
    """Test prediction-to-GT matching."""

    def test_perfect_matching(self):
        """Three predictions perfectly matching three GT."""
        calc = ArcadeMetricsCalculator()

        masks = []
        for i in range(3):
            mask = np.zeros((100, 100), dtype=bool)
            mask[i*30:(i+1)*30, 10:90] = True
            masks.append(mask)

        matches = calc.match_predictions_to_ground_truth(
            masks, masks, iou_threshold=0.5
        )

        assert len(matches) == 3
        for pred_idx, gt_idx, iou in matches:
            assert iou == 1.0

    def test_no_matches_below_threshold(self):
        """No matches when IoU below threshold."""
        calc = ArcadeMetricsCalculator()

        pred_mask = np.zeros((100, 100), dtype=bool)
        pred_mask[0:10, 0:10] = True

        gt_mask = np.zeros((100, 100), dtype=bool)
        gt_mask[90:100, 90:100] = True

        matches = calc.match_predictions_to_ground_truth(
            [pred_mask], [gt_mask], iou_threshold=0.5
        )

        assert len(matches) == 0

    def test_empty_inputs(self):
        """Empty inputs return empty matches."""
        calc = ArcadeMetricsCalculator()

        assert calc.match_predictions_to_ground_truth([], [], 0.5) == []
        assert calc.match_predictions_to_ground_truth([np.zeros((10, 10))], [], 0.5) == []
        assert calc.match_predictions_to_ground_truth([], [np.zeros((10, 10))], 0.5) == []


class TestMetricsForImage:
    """Test full metrics calculation for an image."""

    def test_calculate_metrics_with_matches(self):
        """Test metrics calculation with matching predictions."""
        calc = ArcadeMetricsCalculator()

        # Create prediction annotations
        pred_annotations = [
            {"segmentation": [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]},
        ]

        # Create GT annotations (same as predictions for perfect match)
        gt_annotations = [
            {"segmentation": [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]},
        ]

        metrics = calc.calculate_metrics_for_image(
            pred_annotations,
            gt_annotations,
            image_height=100,
            image_width=100,
            iou_threshold=0.5
        )

        assert metrics["num_matches"] == 1
        assert metrics["num_predictions"] == 1
        assert metrics["num_ground_truth"] == 1
        assert len(metrics["ious"]) == 1
        assert metrics["ious"][0] == 1.0
        assert metrics["dices"][0] == 1.0

    def test_calculate_metrics_no_predictions(self):
        """Test metrics when no predictions."""
        calc = ArcadeMetricsCalculator()

        gt_annotations = [
            {"segmentation": [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]},
        ]

        metrics = calc.calculate_metrics_for_image(
            [],
            gt_annotations,
            image_height=100,
            image_width=100,
            iou_threshold=0.5
        )

        assert metrics["num_matches"] == 0
        assert metrics["num_predictions"] == 0
        assert metrics["num_ground_truth"] == 1