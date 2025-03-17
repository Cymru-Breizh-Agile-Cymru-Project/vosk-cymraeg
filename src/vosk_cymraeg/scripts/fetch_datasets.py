import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.logging import RichHandler

from vosk_cymraeg.datasets.banc_trawsgrifiadau_bangor import (
    fetch_banc_trawsgrifiadau_bangor,
)
from vosk_cymraeg.datasets.common_voice import process_common_voice
from vosk_cymraeg.datasets.lleisiau_arfor import fetch_lleisiau_arfor
from vosk_cymraeg.datasets.enwau_cymraeg import fetch_enwau_cymraeg


@dataclass
class Dataset:
    """A class that holds some information about a dataset"""

    name: str
    output_path: Path
    # A function that takes the target output path
    # and returns nothing
    function: Callable[[Path], None]


# List of available datasets. The keys are used by argparse
DATASETS = {
    "cv": Dataset(
        "Common Voice",
        Path("data/interim/cv/cy"),
        lambda output_path: process_common_voice(Path("data/raw/cv/cy"), output_path),
    ),
    "btb": Dataset(
        "Banc Trawsgrifiadau Bangor",
        Path("data/interim/banc"),
        fetch_banc_trawsgrifiadau_bangor,
    ),
    "enw": Dataset(
        "Enwau Cymraeg",
        Path("data/interim/enwau_cymraeg"),
        fetch_enwau_cymraeg,
    ),
    "lla": Dataset(
        "Lleisiau Arfor",
        Path("data/interim/lleisiau_arfor"),
        fetch_lleisiau_arfor,
    ),
}


def main() -> None:
    """Processes all of the datasets provided in the arguments"""
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    logger = logging.getLogger(__name__)
    args = _get_args()
    console = Console()
    for dataset_id in args.dataset:
        dataset = DATASETS[dataset_id]
        console.rule(f"[bold]{dataset.name}")
        if args.clear:
            logger.warning("Clearning the output folder")
            shutil.rmtree(dataset.output_path)
        dataset.function(dataset.output_path)
        console.line()


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "fetch",
        description="Script responsible for fetching datasets and running the initial processing",
    )

    # Used to wipe the output folder before processing
    parser.add_argument("--clear", action="store_true", help="Clears the target folder")

    # Used to select which datasets to processs. Defaults to all of them
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
    )
    return parser.parse_args()
