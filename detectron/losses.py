"""
Custom loss and metric functions for segmentation tasks.
"""
import torch


def calculate_dice_score(pred_masks, gt_masks, threshold=0.5):
    """
    Calculate DICE score for segmentation masks.

    Args:
        pred_masks: Predicted masks (N, H, W) with values in [0, 1]
        gt_masks: Ground truth masks (N, H, W) with binary values
        threshold: Threshold for binarizing predictions

    Returns:
        Mean DICE score across all masks
    """
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        return torch.tensor(0.0, device=pred_masks.device)

    # Binarize predictions
    pred_binary = (pred_masks > threshold).float()
    gt_binary = gt_masks.float()

    # Calculate DICE per mask
    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
    union = pred_binary.sum(dim=(1, 2)) + gt_binary.sum(dim=(1, 2))

    # DICE = 2 * intersection / (pred + gt)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)

    return dice.mean()


def calculate_iou_score(pred_masks, gt_masks, threshold=0.5):
    """
    Calculate IoU (Intersection over Union) score for segmentation masks.

    Args:
        pred_masks: Predicted masks (N, H, W) with values in [0, 1]
        gt_masks: Ground truth masks (N, H, W) with binary values
        threshold: Threshold for binarizing predictions

    Returns:
        Mean IoU score across all masks
    """
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        return torch.tensor(0.0, device=pred_masks.device)

    # Binarize predictions
    pred_binary = (pred_masks > threshold).float()
    gt_binary = gt_masks.float()

    # Calculate IoU per mask
    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
    union = pred_binary.sum(dim=(1, 2)) + gt_binary.sum(dim=(1, 2)) - intersection

    # IoU = intersection / union
    iou = (intersection + 1e-7) / (union + 1e-7)

    return iou.mean()


def dice_loss(pred_masks, gt_masks):
    """
    Calculate DICE loss for segmentation masks (1 - DICE score).

    Args:
        pred_masks: Predicted masks (N, H, W) with values in [0, 1]
        gt_masks: Ground truth masks (N, H, W) with binary values

    Returns:
        DICE loss value
    """
    dice_score = calculate_dice_score(pred_masks, gt_masks)
    return 1.0 - dice_score


def iou_loss(pred_masks, gt_masks):
    """
    Calculate IoU loss for segmentation masks (1 - IoU score).

    Args:
        pred_masks: Predicted masks (N, H, W) with values in [0, 1]
        gt_masks: Ground truth masks (N, H, W) with binary values

    Returns:
        IoU loss value
    """
    iou_score = calculate_iou_score(pred_masks, gt_masks)
    return 1.0 - iou_score