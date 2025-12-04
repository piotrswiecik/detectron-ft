import os
import cv2
import typer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

app = typer.Typer()

@app.command()
def infer(
        image_path: str,
        model_dir: str = typer.Option(..., help="Path to the directory containing model_final.pth"),
        num_classes: int = typer.Option(1, help="Must match the number of classes used in training"),
        threshold: float = typer.Option(0.5, help="Minimum score threshold to display a prediction"),
        use_cpu: bool = typer.Option(False, help="Force inference on CPU")
):
    """
    Run inference on a single image using a trained Detectron2 model.
    """

    weights_path = os.path.join(model_dir, "model_final.pth")
    if not os.path.exists(weights_path):
        typer.echo(f"Error: Model weights not found at {weights_path}")
        raise typer.Exit(code=1)

    if not os.path.exists(image_path):
        typer.echo(f"Error: Image not found at {image_path}")
        raise typer.Exit(code=1)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    if use_cpu:
        cfg.MODEL.DEVICE = "cpu"

    print(f"Loading model from {weights_path}...")
    predictor = DefaultPredictor(cfg)

    print(f"Processing {image_path}...")
    im = cv2.imread(image_path)
    if im is None:
        typer.echo("Failed to read image.")
        raise typer.Exit(code=1)

    outputs = predictor(im)

    instances = outputs["instances"]
    print(f"Found {len(instances)} detected instances.")

    temp_metadata = MetadataCatalog.get("temp_inference")
    if num_classes == 1:
        temp_metadata.set(thing_classes=["target_object"])
    else:
        temp_metadata.set(thing_classes=[f"class_{i}" for i in range(num_classes)])

    v = Visualizer(im[:, :, ::-1], metadata=temp_metadata, scale=1.0, instance_mode=ColorMode.IMAGE)

    out = v.draw_instance_predictions(instances.to("cpu"))

    result_image = out.get_image()[:, :, ::-1]

    cv2.namedWindow("Inference Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Inference Result", result_image)

    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app()