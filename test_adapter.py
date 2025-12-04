from dataset import Adapter


def test_single_data_point_conversion():
    arcade_json_input = {
        "images": [
            {
                "id": 922,
                "width": 512,
                "height": 512,
                "file_name": "922.png",
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 922,
                "category_id": 8,
                "segmentation": [382.0, 350.75, 390.0, 360.0, 380.0, 370.0],
            }
        ],
    }

    adapter = Adapter(arcade_json_input)

    result = list(adapter)
    item = result[0]

    assert item["image_id"] == 922
    assert len(item["annotations"]) == 1

    ann = item["annotations"][0]

    assert isinstance(ann["segmentation"], list)
    assert isinstance(ann["segmentation"][0], list)

    assert ann["bbox"] == [380.0, 350.75, 390.0, 370.0]
