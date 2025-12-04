import json
import random
import cv2
from detectron2.utils.visualizer import Visualizer

from dataset import Adapter


if __name__ == "__main__":
    path = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/train/annotations/train.json"
    with open(path, "r") as f:
        labels = json.load(f)

    dataset = Adapter(labels, "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/train/images")

    cv2.namedWindow("Detectron2 Visualization", cv2.WINDOW_NORMAL)

    print("Controls: Press any key to see the next image. Press 'ESC' to quit.")

    for d in dataset:
        img = cv2.imread(d["file_name"])

        if img is None:
            print(f"Warning: Could not read {d['file_name']}, skipping...")
            continue

        visualizer = Visualizer(img[:, :, ::-1], metadata=None, scale=0.5)
        out = visualizer.draw_dataset_dict(d)

        result_image_rgb = out.get_image()
        result_image_bgr = result_image_rgb[:, :, ::-1]

        cv2.imshow("Detectron2 Visualization", result_image_bgr)
        print(f"Displaying: {d['file_name']}")

        key = cv2.waitKey(0)

        if key == 27:
            print("Quitting...")
            break

    cv2.destroyAllWindows()

