"""Utility to convert data between Detectron binary mask format and XYXY polygons."""

import torch
import numpy as np
import cv2

def binary_mask_to_xyxy(mask: torch.Tensor, epsilon: float = 0.0):
    mask = mask.detach().to("cpu").numpy()

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask = mask * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if epsilon > 0:
            peri = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon * peri, True)

        if cnt.shape[0] >= 3:
            poly = cnt.reshape(-1, 2).astype(float).tolist()
            poly_flat = [p for xy in poly for p in xy]
            polygons.append(poly_flat)

    return polygons