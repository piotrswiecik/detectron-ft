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