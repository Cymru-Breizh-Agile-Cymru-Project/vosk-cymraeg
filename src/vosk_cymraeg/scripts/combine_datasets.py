import logging
from pathlib import Path

import polars as pl


_logger = logging.getLogger(__name__)


def main():
    """Combines all the predefined splits for the various datasets into a single split
    to create a combined dataset"""
    logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]")

    output_path = Path("data/processed/dataset")
    output_path.mkdir(parents=True, exist_ok=True)
    interim_path = Path("data/interim")

    _logger.info(f"Exporting combined dataset to {output_path}")

    def combine_and_write(name: str, files: list[str]) -> None:
        """Reads the files with polars (paths given relative to the interim folder),
        and exports the combined dataset to the output path with the given name"""
        pl.concat([pl.read_csv(interim_path / file) for file in files]).write_csv(
            output_path / f"{name}.csv"
        )

    # Training set
    combine_and_write(
        "train",
        [
            "banc/train.csv",
            "cv/cy/train.csv",
            "lleisiau_arfor/train_clean.csv",
            "enwau_cymraeg/train.csv",
        ],
    )
    # Validation set
    combine_and_write(
        "dev",
        [
            "banc/validation.csv",
            "cv/cy/dev.csv",
            "lleisiau_arfor/dev_clean.csv",
            "enwau_cymraeg/dev.csv",
        ],
    )
    # Test set
    combine_and_write(
        "test",
        [
            "banc/test.csv",
            "cv/cy/test.csv",
            "lleisiau_arfor/test_clean.csv",
            "enwau_cymraeg/test.csv",
        ],
    )
