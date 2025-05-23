import argparse
import json
import logging
import os
import wave
from pathlib import Path
from typing import Optional

import datasets
import dotenv
import polars as pl
from rich import print
from rich.logging import RichHandler
from tqdm import tqdm
from vosk import KaldiRecognizer, Model

_logger = logging.getLogger(__name__)


def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo


def main() -> None:
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    args = parse_args()

    publish_path: Optional[str] = args.publish_path
    if args.publish:
        _logger.info(
            "After finishing the script will publish the resulting dataset to HuggingFace"
        )
        if publish_path is None:
            publish_path = f"prvInSpace/evals-kaldi-{args.model.name}"
            _logger.warning(
                f"--publish_path not provided. Using default path '{publish_path}'"
            )
            response = input(
                f"Would you like to publish the dataset to '{publish_path}'? [y/N] "
            )
            if response.lower().strip() != "y":
                return

    dataset = pl.read_csv(args.test_data)

    dataset_hash = dataset.hash_rows().sum()
    _logger.info(
        f"The hash of the testing set is '{dataset_hash:x}'. Using this to verify integrity of the results."
    )

    results_path: Path = Path("results/") / f"{args.model.name}_{dataset_hash:x}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    _logger.info(
        f"Model '{args.model.name}' will be tested on '{args.test_data}' and the results will be written to '{results_path}'"
    )

    # Ensure that the file doesn't exist
    if results_path.exists():
        _logger.error(
            f"The file '{results_path}' already exist. To rerun the test, please delete this file"
        )
        return

    print(dataset)
    print("Loading model")
    model = Model(str(args.model))
    with tqdm(desc="Transcribing files", total=len(dataset)) as pbar:
        dataset = dataset.with_columns(
            pl.col("path")
            .map_elements(
                w_pbar(
                    pbar,
                    lambda path: transcribe_file(KaldiRecognizer(model, 16_000), path),
                ),
                pl.String,
            )
            .alias("transcription")
        )

    dataset.write_csv(results_path)

    if args.publish:
        _logger.info(f"Publishing dataset to {publish_path}")
        dotenv.load_dotenv()
        ds = datasets.Dataset(dataset.to_arrow())
        ds.push_to_hub(
            publish_path,
            split="test",
            token=os.environ["HF_TOKEN"],
        )


def transcribe_file(recogniser: KaldiRecognizer, input_path: Path) -> str:
    assert Path(input_path).exists()

    def get_text_from_result(result) -> str:
        result = json.loads(result)
        return result["text"]

    results = []
    with wave.open(input_path) as wf:
        while True:
            data = wf.readframes(8000)
            if len(data) == 0:
                break
            if recogniser.AcceptWaveform(data):
                results.append(get_text_from_result(recogniser.Result()))

    # Need to make sure that there is a result
    final_result = get_text_from_result(recogniser.FinalResult())
    if final_result:
        results.append(final_result)
    recogniser.Reset()
    return " ".join(results).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish_path", type=str)
    return parser.parse_args()
