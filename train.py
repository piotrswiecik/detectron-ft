"""
Module for single training run.
"""

import os
import datetime
import typer
from dotenv import load_dotenv

from detectron.trainer import ArcadeOrchestrator

app = typer.Typer()
load_dotenv()


@app.command()
def run_training(
    arcade_syntax_root: str | None = typer.Option(
        None, help="Path to the root of the arcade-syntax dataset"
    ),
    epochs: int = typer.Option(100, help="Number of epochs"),
    batch: int = typer.Option(2, help="Batch size"),
    base_lr: float = typer.Option(0.001, help="Base learning rate"),
):
    arcade_syntax_root = arcade_syntax_root or os.getenv("ARCADE_SYNTAX_ROOT")
    if arcade_syntax_root is None:
        typer.echo("ARCADE_SYNTAX_ROOT not set.")
        raise typer.Exit(code=1)
    tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"training_{tstamp}")
    os.makedirs(output_path, exist_ok=True)
    orchestrator = ArcadeOrchestrator(
        arcade_syntax_root, model_output_dir=output_path
    )
    orchestrator.train(epochs, batch, base_lr)


if __name__ == "__main__":
    app()
