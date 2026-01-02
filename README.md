# Detectron2 training

```shell
uv add torch torchvision
# or
uv sync
# and
uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

Training

```shell
uv run train.py <data-path> <out-dir> <epochs>
`````

Inference test

```shell
uv run sample.py /Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/test/images/1.png --num-classes 25 --model-dir=output --use-cpu --threshold 0.2 
```

```shell
uv run bulk_detect.py --image-root /Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/ --out-root /Users/piotrswiecik/dev/ives/coronary/testing/20251208_111315_lr001_freeze0 --num-classes 25 --model-dir=/Users/piotrswiecik/dev/ives/coronary/trained_models/detectron/20251208_111315_lr001_freeze0 --use-cpu --threshold 0.2
```