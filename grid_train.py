"""
Module for grid training.
"""

import os
import datetime
import typer
import itertools
from dotenv import load_dotenv

from detectron.trainer import ArcadeOrchestrator

app = typer.Typer()
load_dotenv()


@app.command()
def run_grid(
    arcade_syntax_root: str | None = typer.Option(
        None, help="Path to the root of the arcade-syntax dataset"
    ),
    epochs: int = typer.Option(100, help="Number of epochs"),
    batch: int = typer.Option(2, help="Batch size"),
):
    arcade_syntax_root = arcade_syntax_root or os.getenv("ARCADE_SYNTAX_ROOT")
    if arcade_syntax_root is None:
        typer.echo("ARCADE_SYNTAX_ROOT not set.")
        raise typer.Exit(code=1)

    LR_GRID_PARAMS = [0.00025, 0.0005, 0.001, 0.0025]
    ANCHOR_SIZES = [
        [[16], [32], [64], [128], [256]],
        [[32], [64], [128], [256], [512]],
    ]

    ANCHOR_ASPECT_RATIOS = [
        [[0.5, 1.0, 2.0]],
        [[0.33, 0.5, 1.0, 2.0, 3.0]],
        [[0.75, 1.0, 1.5]],
    ]
    BACKBONE_FREEZE_AT = [0, 1, 2]

    param_combinations = list(
        itertools.product(
            LR_GRID_PARAMS, ANCHOR_SIZES, ANCHOR_ASPECT_RATIOS, BACKBONE_FREEZE_AT
        )
    )

    for params in param_combinations:
        tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        typer.echo(f"Training with params: {params} at {tstamp}")
        orchestrator = ArcadeOrchestrator(
            arcade_syntax_root, model_output_dir=f"training_{tstamp}"
        )
        lr = params[0]
        params_dict = {
            "anchor_sizes": params[1],
            "anchor_ratios": params[2],
            "freeze_at": params[3],
        }
        orchestrator.train(epochs, batch, lr, params_dict)


if __name__ == "__main__":
    app()
