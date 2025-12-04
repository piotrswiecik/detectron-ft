import os
from collections import defaultdict
from typing import TypeAlias, TypedDict, Iterator

from detectron2.structures import keypoints
from pydantic import BaseModel

"""
Shape of ARCADE annotation format.
{
  "images": [
    {
      "id": 922,
      "width": 512,
      "height": 512,
      "file_name": "922.png",
      "license": 0,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": 0
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 922,
      "category_id": 8,
      "segmentation": [
        382.0,
        350.75,
      ]
    }
  ]
}

Shape of Detectron annotation format.
{
    "file_name": "path",
    "height": 100,
    "width": 100,
    "image_id": 1,
    "annotations": [
        {
            "bbox": [x1, y1, width, height],
            "bbox_mode": 0, 
            "category_id": 1,
            "segmentation": [[x1, y1, ..., xn, yn], ... [x1, y1, ..., xn, yn]],
            "keypoints": [x1, y1, v1, ..., xn, yn, vn]
        }
}

Bbox modes: XYXY_ABS = 0, XYWH_ABS = 1, XYXY_REL = 2, XYWH_REL = 3, XYWHA_ABS = 4
"""


class Adapter:
    """
    Dataset adapter from ARCADE annotation format to Detectron format.
    """
    def __init__(self, arcade: dict, image_root: str):
        self.images = [img for img in arcade.get("images", [])]
        self._raw_anns = [ann for ann in arcade.get("annotations", [])]
        self._grouped_anns = defaultdict(list)
        for ann in self._raw_anns:
            self._grouped_anns[ann["image_id"]].append(ann)
        self._box_mode = 0  # XYXY_ABS
        self._image_root = image_root

    @staticmethod
    def _calculate_xyxyabs_bbox(segmentation: list) -> list[float]:
        """
        Calculates bbox from either a flat list [x,y,x,y] or nested list [[x,y...], [x,y...]].
        """
        if not segmentation:
            return [0.0, 0.0, 0.0, 0.0]

        if isinstance(segmentation[0], list):
            flat_coords = [coord for polygon in segmentation for coord in polygon]
        else:
            flat_coords = segmentation

        if not flat_coords:
            return [0.0, 0.0, 0.0, 0.0]

        xs = flat_coords[0::2]
        ys = flat_coords[1::2]

        return [min(xs), min(ys), max(xs), max(ys)]

    def _convert_annotation(self, ann: dict) -> dict:
        raw_seg = ann["segmentation"]

        bbox = self._calculate_xyxyabs_bbox(raw_seg)

        # If input is [x, y...], wrap it -> [[x, y...]]
        # If input is [[x, y...]], keep it -> [[x, y...]]
        if raw_seg and not isinstance(raw_seg[0], list):
            final_seg = [raw_seg]
        else:
            final_seg = raw_seg

        return {
            "bbox": bbox,
            "bbox_mode": self._box_mode,
            "category_id": ann["category_id"],
            "segmentation": final_seg,
            "keypoints": [],
        }

    def __iter__(self) -> Iterator[dict]:
        for img in self.images:
            related_anns = self._grouped_anns.get(img["id"], [])
            converted_anns = [self._convert_annotation(ann) for ann in related_anns]
            yield {
                "file_name": os.path.join(self._image_root, img["file_name"]),
                "height": img["height"],
                "width": img["width"],
                "image_id": img["id"],
                "annotations": converted_anns,
            }

    def as_list(self) -> list[dict]:
        return list(self)
